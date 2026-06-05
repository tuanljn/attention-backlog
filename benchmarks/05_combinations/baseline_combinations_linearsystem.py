# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Combination benchmark on LinearSystem — tests Donon's "building block" hypothesis.

Tests whether combining `GlobalAggregationMessagePassingFunction` (global context
layer) with a local message function (`LocalSumMessagePassingFunction` or
`GATv2MessagePassingFunction`) inside a single `RecurrentCoupler` outperforms
the local message function alone.

The composition mechanism is the existing `RecurrentCoupler` with a list of
message functions: each step calls both message functions, concatenates the
outputs along the feature axis, then applies `phi`. This is the simplest form
of composition, available without writing new coupler code, and provides an
empirical lower bound for the future `VirtualAddressRecurrentCoupler` (Item 5
of `attention-backlog.md`).

Two combinations tested:
    - LocalSum + GlobalAggregation
    - GATv2 + GlobalAggregation

Dataset config / seeds / epoch budgets match `baseline_linearsystem` so the
results are directly comparable against the standalone baselines.

A `--smoke` flag runs a single seed × single size × single epoch for a
sanity-check pass before the full canonical run. See `--help` for usage.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from energnn.model import (
    GATv2MessagePassingFunction,
    GlobalAggregationMessagePassingFunction,
    GNN,
    LocalSumMessagePassingFunction,
    MLP,
    MLPEncoder,
    MLPEquivariantDecoder,
    RecurrentCoupler,
    TDigestNormalizer,
)
from energnn.problem.example import LinearSystemProblemLoader
from energnn.trainer import Trainer

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "results" / HERE.name

# Matched to baseline_linearsystem.py for apples-to-apples comparison.
N_MAX = 3
DATASET_SIZE = 64
BATCH_SIZE = 4
VAL_DATASET_SIZE = 32
SEEDS = (0, 1, 2)


@dataclass
class SizeConfig:
    name: str
    n_breakpoints: int
    latent_dim: int
    hidden_sizes: tuple[int, ...]
    n_steps: int
    n_seeds: int
    n_epochs: int


SIZE_CONFIGS = (
    SizeConfig(name="Tiny",  n_breakpoints=10, latent_dim=4, hidden_sizes=(),    n_steps=5,  n_seeds=3, n_epochs=10),
    SizeConfig(name="Small", n_breakpoints=20, latent_dim=8, hidden_sizes=(16,), n_steps=10, n_seeds=3, n_epochs=15),
)


# Two compositions tested. Each entry is a callable that builds the two message
# functions to compose. The order matters only for output concatenation order,
# which is symmetric under `phi`.
COMBOS = {
    "LocalSum+GlobalAgg": "local_global",
    "GATv2+GlobalAgg":    "gatv2_global",
}


def build_combo_gnn(
    config: SizeConfig,
    in_structure,
    out_structure,
    *,
    seed: int,
    combo: str,
) -> GNN:
    """Build a GNN whose coupler combines two message functions.

    The two messages run in parallel per step, their outputs are concatenated
    along the feature axis, and `phi` reduces the concatenation back to
    `latent_dim`. This is the existing `RecurrentCoupler` behaviour with a
    list of message functions.

    :param combo: one of `local_global` or `gatv2_global`.
    """
    rngs = nnx.Rngs(seed)
    normalizer = TDigestNormalizer(
        in_structure=in_structure,
        n_breakpoints=config.n_breakpoints,
        update_limit=1000,
    )
    encoder = MLPEncoder(
        in_structure=in_structure,
        hidden_sizes=list(config.hidden_sizes),
        activation=nnx.leaky_relu,
        out_size=config.latent_dim,
        use_bias=True,
        final_activation=None,
        rngs=rngs,
    )

    # Build the two message functions. The local one always produces output
    # of size `latent_dim`; the global one likewise. So the concat has feature
    # dimension `2 * latent_dim`, and `phi.in_size` must match.
    if combo == "local_global":
        msg_local = LocalSumMessagePassingFunction(
            in_graph_structure=in_structure,
            in_array_size=config.latent_dim,
            hidden_sizes=list(config.hidden_sizes),
            activation=nnx.leaky_relu,
            out_size=config.latent_dim,
            use_bias=True,
            final_activation=None,
            outer_activation=nnx.tanh,
            encoded_feature_size=config.latent_dim,
            rngs=rngs,
        )
    elif combo == "gatv2_global":
        msg_local = GATv2MessagePassingFunction(
            in_graph_structure=in_structure,
            in_array_size=config.latent_dim,
            hidden_sizes=list(config.hidden_sizes),
            activation=nnx.leaky_relu,
            out_size=config.latent_dim,
            use_bias=True,
            final_activation=None,
            outer_activation=nnx.tanh,
            encoded_feature_size=config.latent_dim,
            rngs=rngs,
        )
    else:
        raise ValueError(f"unknown combo: {combo}")

    msg_global = GlobalAggregationMessagePassingFunction(
        in_graph_structure=in_structure,
        in_array_size=config.latent_dim,
        hidden_sizes=list(config.hidden_sizes),
        activation=nnx.leaky_relu,
        out_size=config.latent_dim,
        use_bias=True,
        final_activation=None,
        outer_activation=nnx.tanh,
        rngs=rngs,
    )

    # phi receives the concatenation of the two messages, hence in_size doubled.
    phi = MLP(
        in_size=2 * config.latent_dim,
        hidden_sizes=[],
        activation=nnx.leaky_relu,
        out_size=config.latent_dim,
        use_bias=True,
        final_activation=nnx.tanh,
        rngs=rngs,
    )
    coupler = RecurrentCoupler(
        phi=phi,
        message_functions=[msg_local, msg_global],
        n_steps=config.n_steps,
    )
    decoder = MLPEquivariantDecoder(
        in_graph_structure=in_structure,
        in_array_size=config.latent_dim,
        hidden_sizes=list(config.hidden_sizes),
        activation=nnx.leaky_relu,
        out_structure=out_structure,
        use_bias=True,
        final_activation=None,
        encoded_feature_size=config.latent_dim,
        rngs=rngs,
    )
    return GNN(normalizer=normalizer, encoder=encoder, coupler=coupler, decoder=decoder)


@dataclass
class RunResult:
    combo: str
    size: str
    seed: int
    n_epochs: int
    n_params: int
    eval_before: float
    eval_after: float
    eval_improvement: float
    epoch_eval_curve: list[float]
    median_step_time_ms: float
    total_train_time_s: float
    peak_memory_mb: float


@dataclass
class ComboSizeSummary:
    combo: str
    size: str
    n_seeds: int
    n_params: int
    eval_after_median: float
    best_eval_median: float
    median_step_time_ms: float


@dataclass
class BenchmarkReport:
    env: dict
    config: dict
    runs: list[dict] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)


def env_fingerprint() -> dict:
    import flax as _flax
    import optax as _optax
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "flax": _flax.__version__,
        "optax": _optax.__version__,
    }


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024


def count_params(model) -> int:
    _, params, _ = nnx.split(model, nnx.Param, ...)
    total = 0
    for leaf in jax.tree_util.tree_leaves(params):
        total += int(np.prod(leaf.shape))
    return total


def train_one_run(
    combo: str,
    config: SizeConfig,
    seed: int,
    n_epochs: int,
) -> RunResult:
    """Train a single (combo, size, seed) configuration. Returns the result.

    Same data pipeline and optimizer as `baseline_linearsystem` — the only
    differences are the GNN's coupler is a 2-message composition and
    `phi.in_size = 2 * latent_dim`.
    """
    print(f"  [{combo:<22s}] {config.name:<5s} seed={seed} n_epochs={n_epochs} ...", end=" ", flush=True)
    sys.stdout.flush()

    train_loader = LinearSystemProblemLoader(
        seed=10 * seed + 7,
        n_max=N_MAX,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_loader = LinearSystemProblemLoader(
        seed=10 * seed + 8,
        n_max=N_MAX,
        dataset_size=VAL_DATASET_SIZE,
        batch_size=BATCH_SIZE,
    )

    gnn = build_combo_gnn(
        config=config,
        in_structure=train_loader.context_structure,
        out_structure=train_loader.decision_structure,
        seed=seed,
        combo=COMBOS[combo],
    )
    n_params = count_params(gnn)

    trainer = Trainer(model=gnn, gradient_transformation=optax.adam(1e-3))
    eval_before, _ = trainer.eval(val_loader, progress_bar=False)
    eval_before = float(eval_before)

    step_times: list[float] = []
    curve: list[float] = [eval_before]
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        step_t0 = time.perf_counter()
        trainer.train(
            train_loader=train_loader,
            val_loader=None,
            n_epochs=1,
            progress_bar=False,
            eval_before_training=False,
            eval_after_epoch=False,
        )
        step_times.append((time.perf_counter() - step_t0) * 1000.0)
        eval_now, _ = trainer.eval(val_loader, progress_bar=False)
        curve.append(float(eval_now))
    total_train_time = time.perf_counter() - t0

    median_step_ms = float(statistics.median(step_times)) if step_times else 0.0
    eval_after = float(curve[-1])
    peak_mem = peak_memory_mb()

    result = RunResult(
        combo=combo,
        size=config.name,
        seed=seed,
        n_epochs=n_epochs,
        n_params=n_params,
        eval_before=eval_before,
        eval_after=eval_after,
        eval_improvement=eval_before - eval_after,
        epoch_eval_curve=curve,
        median_step_time_ms=median_step_ms,
        total_train_time_s=float(total_train_time),
        peak_memory_mb=peak_mem,
    )
    print(f"eval {eval_before:.3e} -> {eval_after:.3e}  ({median_step_ms:.1f} ms/step, {total_train_time:.1f}s, n_params={n_params})")
    return result


def aggregate_summaries(runs: list[RunResult]) -> list[ComboSizeSummary]:
    """Median statistics over seeds per (combo, size)."""
    summaries: list[ComboSizeSummary] = []
    keys = {(r.combo, r.size) for r in runs}
    for combo, size in sorted(keys):
        block = [r for r in runs if r.combo == combo and r.size == size]
        eval_afters = [r.eval_after for r in block]
        bests = [min(r.epoch_eval_curve) for r in block]
        step_ms = [r.median_step_time_ms for r in block]
        summaries.append(
            ComboSizeSummary(
                combo=combo,
                size=size,
                n_seeds=len(block),
                n_params=block[0].n_params,
                eval_after_median=float(statistics.median(eval_afters)),
                best_eval_median=float(statistics.median(bests)),
                median_step_time_ms=float(statistics.median(step_ms)),
            )
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: run only Tiny / seed 0 / 1 epoch / first combo, to verify pipeline works.",
    )
    parser.add_argument(
        "--combo",
        choices=list(COMBOS.keys()) + ["all"],
        default="all",
        help="Which combo(s) to run. Default: all.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        sizes = (SIZE_CONFIGS[0],)
        seeds = (0,)
        n_epochs_override = 1
        combos_to_run = (list(COMBOS.keys())[0],) if args.combo == "all" else (args.combo,)
        out_path = RESULTS_DIR / "smoke_combinations_linearsystem.json"
        print(f"=== SMOKE TEST MODE: 1 combo / 1 size / 1 seed / 1 epoch ===")
    else:
        sizes = SIZE_CONFIGS
        seeds = SEEDS
        n_epochs_override = None
        combos_to_run = tuple(COMBOS.keys()) if args.combo == "all" else (args.combo,)
        out_path = RESULTS_DIR / "baseline_combinations_linearsystem.json"
        print(f"=== FULL RUN: {len(combos_to_run)} combos × {len(sizes)} sizes × {len(seeds)} seeds = {len(combos_to_run) * len(sizes) * len(seeds)} runs ===")

    print(f"  combos: {combos_to_run}")
    print(f"  sizes:  {[c.name for c in sizes]}")
    print(f"  seeds:  {seeds}")
    print()

    runs: list[RunResult] = []
    for combo in combos_to_run:
        for cfg in sizes:
            for seed in seeds:
                n_epochs = n_epochs_override if n_epochs_override else cfg.n_epochs
                result = train_one_run(combo=combo, config=cfg, seed=seed, n_epochs=n_epochs)
                runs.append(result)
                gc.collect()

    summaries = aggregate_summaries(runs)

    print()
    print("== Per-(combo, size) summary ==")
    print(f"  {'combo':<22s} {'size':<6s} {'n_params':>9s} {'final-eval (med)':>18s} {'best-eval (med)':>18s} {'step (ms)':>11s}")
    print("  " + "-" * 90)
    for s in summaries:
        print(f"  {s.combo:<22s} {s.size:<6s} {s.n_params:>9d} {s.eval_after_median:>18.3e} {s.best_eval_median:>18.3e} {s.median_step_time_ms:>11.1f}")

    report = BenchmarkReport(
        env=env_fingerprint(),
        config={
            "dataset": "LinearSystem",
            "n_max": N_MAX,
            "dataset_size": DATASET_SIZE,
            "batch_size": BATCH_SIZE,
            "val_dataset_size": VAL_DATASET_SIZE,
            "combos": list(combos_to_run),
            "sizes": [c.name for c in sizes],
            "seeds": list(seeds),
            "smoke_mode": args.smoke,
        },
        runs=[asdict(r) for r in runs],
        summaries=[asdict(s) for s in summaries],
    )
    out_path.write_text(json.dumps(asdict(report), indent=2))
    print()
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()

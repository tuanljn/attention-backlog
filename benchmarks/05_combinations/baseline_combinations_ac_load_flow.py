# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""Combination benchmark on IEEE AC load flow — tests "building block" hypothesis on realistic topology.

Two compositions tested inside a single `RecurrentCoupler`:
    - `LocalSum + GlobalAggregation`
    - `GATv2 + GlobalAggregation`

Same `ACLoadFlowProblemLoader` config, same seeds, same epoch budgets as
`baseline_gatv2_ac_load_flow.py` so the results are directly comparable
against the standalone baselines from Approches 1, 2, 3 and the LocalSum
reference.

The composition mechanism is the standard `RecurrentCoupler` with a list of
two message functions. Each step: both messages run in parallel, outputs are
concatenated along the feature axis, then `phi` reduces back to `latent_dim`.
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
from energnn.trainer import Trainer

# Import the AC LF problem loader from the benchmarks package alongside this file.
sys.path.insert(0, str(Path(__file__).parent.parent))
from ac_load_flow_problem import ACLoadFlowProblemLoader  # noqa: E402

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "results" / HERE.name

IEEE_SIZES = (9, 14)
PERTURBATION_SCALE = 0.1
DATASET_SIZE = 32
BATCH_SIZE = 4
VAL_DATASET_SIZE = 16
SEEDS = (0, 1, 2)


@dataclass
class SizeConfig:
    name: str
    n_breakpoints: int
    latent_dim: int
    hidden_sizes: tuple[int, ...]
    n_steps: int
    n_epochs: int


SIZE_CONFIGS = (
    SizeConfig(name="Tiny",  n_breakpoints=10, latent_dim=4, hidden_sizes=(),    n_steps=5,  n_epochs=10),
    SizeConfig(name="Small", n_breakpoints=20, latent_dim=8, hidden_sizes=(16,), n_steps=10, n_epochs=15),
)


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
    """Build a GNN whose RecurrentCoupler composes two message functions."""
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
    network: str
    size: str
    seed: int
    n_epochs: int
    n_params: int
    eval_before: float
    eval_after: float
    epoch_eval_curve: list[float]
    median_step_time_ms: float
    total_train_time_s: float
    peak_memory_mb: float


@dataclass
class BenchmarkReport:
    env: dict
    config: dict
    runs: list[dict] = field(default_factory=list)


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
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def count_params(model) -> int:
    _, params, _ = nnx.split(model, nnx.Param, ...)
    total = 0
    for leaf in jax.tree_util.tree_leaves(params):
        total += int(np.prod(leaf.shape))
    return total


def run_one(
    combo: str,
    network: str,
    config: SizeConfig,
    seed: int,
    n_epochs: int,
) -> RunResult:
    """Train one (combo, network, size, seed) run on the IEEE AC LF substrate."""
    train_loader = ACLoadFlowProblemLoader(
        network_name=network,
        seed=10 * seed + 7,
        perturbation_scale=PERTURBATION_SCALE,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_loader = ACLoadFlowProblemLoader(
        network_name=network,
        seed=10 * seed + 8,
        perturbation_scale=PERTURBATION_SCALE,
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

    return RunResult(
        combo=combo,
        network=network,
        size=config.name,
        seed=seed,
        n_epochs=n_epochs,
        n_params=n_params,
        eval_before=eval_before,
        eval_after=float(curve[-1]),
        epoch_eval_curve=curve,
        median_step_time_ms=float(statistics.median(step_times)) if step_times else 0.0,
        total_train_time_s=float(total_train_time),
        peak_memory_mb=peak_memory_mb(),
    )


def write_partial(all_runs: list[RunResult], smoke: bool) -> Path:
    out_path = RESULTS_DIR / ("smoke_combinations_ac_load_flow.json" if smoke else "baseline_combinations_ac_load_flow.json")
    report = BenchmarkReport(
        env=env_fingerprint(),
        config={
            "ieee_sizes": list(IEEE_SIZES),
            "perturbation_scale": PERTURBATION_SCALE,
            "dataset_size": DATASET_SIZE,
            "batch_size": BATCH_SIZE,
            "val_dataset_size": VAL_DATASET_SIZE,
            "seeds": list(SEEDS),
            "sizes": [{"name": c.name, "n_epochs": c.n_epochs} for c in SIZE_CONFIGS],
            "combos": list(COMBOS.keys()),
            "smoke_mode": smoke,
        },
        runs=[asdict(r) for r in all_runs],
    )
    out_path.write_text(json.dumps(asdict(report), indent=2, default=float))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke: 1 combo × 1 network (ieee9) × 1 size (Tiny) × 1 seed × 1 epoch.",
    )
    parser.add_argument(
        "--combo",
        choices=list(COMBOS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        combos_to_run = (list(COMBOS.keys())[0],) if args.combo == "all" else (args.combo,)
        sizes_to_run = (SIZE_CONFIGS[0],)
        seeds_to_run = (0,)
        ieee_sizes_to_run = (9,)
        n_epochs_override = 1
        print("=== SMOKE TEST MODE: AC LF, 1 combo × 1 network × 1 size × 1 seed × 1 epoch ===")
    else:
        combos_to_run = tuple(COMBOS.keys()) if args.combo == "all" else (args.combo,)
        sizes_to_run = SIZE_CONFIGS
        seeds_to_run = SEEDS
        ieee_sizes_to_run = IEEE_SIZES
        n_epochs_override = None
        total = len(combos_to_run) * len(ieee_sizes_to_run) * len(sizes_to_run) * len(seeds_to_run)
        print(f"== Combinations on IEEE AC LF ({total} runs) ==")

    print(f"  combos: {combos_to_run}")
    print(f"  IEEE sizes: {ieee_sizes_to_run}")
    print(f"  sizes: {[c.name for c in sizes_to_run]}")
    print(f"  seeds: {seeds_to_run}")
    print(f"  perturbation_scale={PERTURBATION_SCALE}, train={DATASET_SIZE}, batch={BATCH_SIZE}, val={VAL_DATASET_SIZE}")
    print()

    all_runs: list[RunResult] = []
    total = len(combos_to_run) * len(ieee_sizes_to_run) * len(sizes_to_run) * len(seeds_to_run)
    idx = 0
    overall_t0 = time.perf_counter()

    for combo in combos_to_run:
        for size_int in ieee_sizes_to_run:
            network = f"ieee{size_int}"
            for cfg in sizes_to_run:
                for seed in seeds_to_run:
                    idx += 1
                    n_ep = n_epochs_override if n_epochs_override else cfg.n_epochs
                    print(
                        f"  [{idx:2d}/{total}] {combo:<22s} {network:<6s} {cfg.name:<5s} "
                        f"seed={seed} n_epochs={n_ep} ... ",
                        end="",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    try:
                        result = run_one(combo=combo, network=network, config=cfg, seed=seed, n_epochs=n_ep)
                        all_runs.append(result)
                        elapsed = time.perf_counter() - t0
                        print(
                            f"eval {result.eval_before:.3e} -> {result.eval_after:.3e}  "
                            f"({result.median_step_time_ms:.1f} ms/step, {elapsed:.1f}s, n_params={result.n_params})"
                        )
                    except Exception as exc:
                        elapsed = time.perf_counter() - t0
                        print(f"FAIL after {elapsed:.1f}s -- {type(exc).__name__}: {exc}")
                    gc.collect()
                    write_partial(all_runs, smoke=args.smoke)

    print()
    print("== Per-(combo, network, size) summary ==")
    print(f"  {'combo':22s} {'network':8s} {'size':6s} {'n_params':>9s} {'best_eval (med)':>17s} {'step_ms':>9s}")
    print("  " + "-" * 80)
    keys = {(r.combo, r.network, r.size) for r in all_runs}
    for combo, net, size in sorted(keys):
        block = [r for r in all_runs if r.combo == combo and r.network == net and r.size == size]
        bests = [min(r.epoch_eval_curve) for r in block]
        steps = [r.median_step_time_ms for r in block]
        print(
            f"  {combo:22s} {net:8s} {size:6s} {block[0].n_params:>9d} "
            f"{statistics.median(bests):>17.3e} {statistics.median(steps):>9.1f}"
        )

    overall = time.perf_counter() - overall_t0
    print(f"\nTotal elapsed: {overall:.1f}s ({overall/60:.1f} min)")
    print(f"Results: {RESULTS_DIR / ('smoke_combinations_ac_load_flow.json' if args.smoke else 'baseline_combinations_ac_load_flow.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

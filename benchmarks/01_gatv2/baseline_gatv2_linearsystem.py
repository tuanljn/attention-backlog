# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
GATv2 baseline on LinearSystem — apples-to-apples comparison vs LocalSum.

Mirrors :mod:`baseline_linearsystem` (same dataset config, same seeds, same
epoch budgets, same optimizer) but swaps :class:`LocalSumMessagePassingFunction`
for :class:`GATv2MessagePassingFunction` inside the ``RecurrentCoupler``. The
rest of the pipeline (normalizer, encoder, decoder, optimizer) is identical
to ``ReadyRecurrentEquivariantGNN``.

Output JSON consumed by :mod:`render_baselines_md` to extend BASELINES.md
with a side-by-side comparison table for Item 1 (GATv2) of
``attention-backlog.md``.
"""
from __future__ import annotations

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
    GNN,
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


# Mirrors Tiny / Small in energnn.model.ready_to_use, but the message function
# is swapped to GATv2. Larger sizes (Medium / Large / ExtraLarge) deliberately
# excluded to match baseline_linearsystem.py's reduced scope.
SIZE_CONFIGS = (
    SizeConfig(name="Tiny",  n_breakpoints=10, latent_dim=4, hidden_sizes=(),    n_steps=5,  n_seeds=3, n_epochs=10),
    SizeConfig(name="Small", n_breakpoints=20, latent_dim=8, hidden_sizes=(16,), n_steps=10, n_seeds=3, n_epochs=15),
)


def build_gatv2_gnn(config: SizeConfig, in_structure, out_structure, *, seed: int) -> GNN:
    """Build a GATv2-equipped GNN matching the ready-to-use config sizes.

    Identical to ``ReadyRecurrentEquivariantGNN`` except the message function
    is :class:`GATv2MessagePassingFunction`.
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
    message_function = GATv2MessagePassingFunction(
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
    phi = MLP(
        in_size=config.latent_dim,
        hidden_sizes=[],
        activation=nnx.leaky_relu,
        out_size=config.latent_dim,
        use_bias=True,
        final_activation=nnx.tanh,
        rngs=rngs,
    )
    coupler = RecurrentCoupler(
        phi=phi,
        message_functions=[message_function],
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
    size: str
    seed: int
    n_epochs: int
    n_params: int
    eval_before: float
    eval_after: float
    eval_improvement: float
    epoch_eval_curve: list[float]
    median_step_time_ms: float
    p90_step_time_ms: float
    total_train_time_s: float
    peak_memory_mb: float
    warning: str = ""


@dataclass
class SizeSummary:
    size: str
    n_seeds: int
    n_params: int
    eval_after_min: float
    eval_after_median: float
    eval_after_max: float
    best_eval_median: float
    eval_improvement_median: float
    median_step_time_ms: float
    peak_memory_mb_median: float


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
    return usage / 1024  # ru_maxrss is kilobytes on Linux


def count_params(model) -> int:
    _, params, _ = nnx.split(model, nnx.Param, ...)
    total = 0
    for leaf in jax.tree_util.tree_leaves(params):
        if hasattr(leaf, "size"):
            total += int(leaf.size)
    return total


def measure_step_times(trainer, train_loader, n_steps: int = 30) -> tuple[float, float]:
    for problem_batch in train_loader:
        _ = trainer.training_step(problem_batch, get_info=False)
        break
    step_times: list[float] = []
    while len(step_times) < n_steps:
        for problem_batch in train_loader:
            t0 = time.perf_counter()
            _ = trainer.training_step(problem_batch, get_info=False)
            step_times.append((time.perf_counter() - t0) * 1000.0)
            if len(step_times) >= n_steps:
                break
    median = statistics.median(step_times)
    p90 = statistics.quantiles(step_times, n=10)[-1] if len(step_times) >= 10 else max(step_times)
    return median, p90


def eval_score(trainer, val_loader) -> float:
    score, _ = trainer.eval(val_loader, progress_bar=False)
    return float(score)


def run_one(config: SizeConfig, seed: int) -> RunResult:
    warning = ""
    train_loader = LinearSystemProblemLoader(
        seed=10 * seed + 7,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        n_max=N_MAX,
    )
    val_loader = LinearSystemProblemLoader(
        seed=10 * seed + 8,
        dataset_size=VAL_DATASET_SIZE,
        batch_size=BATCH_SIZE,
        n_max=N_MAX,
    )

    model = build_gatv2_gnn(
        config,
        in_structure=train_loader.context_structure,
        out_structure=train_loader.decision_structure,
        seed=seed,
    )
    n_params = count_params(model)

    trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-3))

    eval_before = eval_score(trainer, val_loader)
    if not np.isfinite(eval_before):
        warning = f"non-finite eval_before={eval_before}"

    median_ms, p90_ms = measure_step_times(trainer, train_loader, n_steps=20)

    epoch_eval_curve: list[float] = []
    t0 = time.perf_counter()
    for _ in range(config.n_epochs):
        trainer.train(
            train_loader=train_loader,
            val_loader=None,
            n_epochs=1,
            progress_bar=False,
            eval_before_training=False,
            eval_after_epoch=False,
        )
        epoch_eval_curve.append(eval_score(trainer, val_loader))
    total_time_s = time.perf_counter() - t0

    eval_after = epoch_eval_curve[-1] if epoch_eval_curve else eval_before
    eval_improvement = (
        (eval_before - eval_after) if np.isfinite(eval_before) and np.isfinite(eval_after) else float("nan")
    )
    if not np.isfinite(eval_after):
        warning = (warning + "; " if warning else "") + f"non-finite eval_after={eval_after}"

    return RunResult(
        size=config.name,
        seed=seed,
        n_epochs=config.n_epochs,
        n_params=n_params,
        eval_before=eval_before,
        eval_after=eval_after,
        eval_improvement=eval_improvement,
        epoch_eval_curve=epoch_eval_curve,
        median_step_time_ms=median_ms,
        p90_step_time_ms=p90_ms,
        total_train_time_s=total_time_s,
        peak_memory_mb=peak_memory_mb(),
        warning=warning,
    )


def summarise(runs: list[RunResult]) -> list[SizeSummary]:
    by_size: dict[str, list[RunResult]] = {}
    for r in runs:
        by_size.setdefault(r.size, []).append(r)
    out = []
    for size, rs in by_size.items():
        finals = [r.eval_after for r in rs if np.isfinite(r.eval_after)] or [float("nan")]
        bests = []
        for r in rs:
            curve = [v for v in r.epoch_eval_curve if np.isfinite(v)]
            if curve:
                bests.append(min(curve))
        if not bests:
            bests = [float("nan")]
        improvements = [r.eval_improvement for r in rs if np.isfinite(r.eval_improvement)]
        median_imp = statistics.median(improvements) if improvements else float("nan")
        out.append(
            SizeSummary(
                size=size,
                n_seeds=len(rs),
                n_params=rs[0].n_params,
                eval_after_min=min(finals),
                eval_after_median=statistics.median(finals),
                eval_after_max=max(finals),
                best_eval_median=statistics.median(bests),
                eval_improvement_median=median_imp,
                median_step_time_ms=statistics.median(r.median_step_time_ms for r in rs),
                peak_memory_mb_median=statistics.median(r.peak_memory_mb for r in rs),
            )
        )
    return out


def write_partial(all_runs: list[RunResult], status: str) -> Path:
    report = BenchmarkReport(
        env=env_fingerprint(),
        config={
            "n_max": N_MAX,
            "dataset_size": DATASET_SIZE,
            "batch_size": BATCH_SIZE,
            "val_dataset_size": VAL_DATASET_SIZE,
            "seeds": list(SEEDS),
            "sizes": [
                {"name": c.name, "n_seeds": c.n_seeds, "n_epochs": c.n_epochs}
                for c in SIZE_CONFIGS
            ],
            "message_function": "GATv2MessagePassingFunction",
            "optimizer": "optax.adam(1e-3)",
            "status": status,
            "compared_against": "baseline_linearsystem.json (LocalSumMessagePassingFunction)",
        },
        runs=[asdict(r) for r in all_runs],
        summaries=[asdict(s) for s in summarise(all_runs)],
    )
    out_path = RESULTS_DIR / "baseline_gatv2_linearsystem.json"
    out_path.write_text(json.dumps(asdict(report), indent=2, default=float))
    return out_path


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("== GATv2 baseline on LinearSystem (apples-to-apples vs LocalSum) ==")
    print(f"  n_max={N_MAX}, train={DATASET_SIZE}, batch={BATCH_SIZE}, val={VAL_DATASET_SIZE}")
    print(f"  sizes={[c.name for c in SIZE_CONFIGS]}, seeds={SEEDS}")
    print()

    all_runs: list[RunResult] = []
    total = sum(c.n_seeds for c in SIZE_CONFIGS)
    idx = 0
    t0_overall = time.perf_counter()
    for config in SIZE_CONFIGS:
        for seed in SEEDS[: config.n_seeds]:
            idx += 1
            print(
                f"  [{idx:2d}/{total}] {config.name:<5s} seed={seed} n_epochs={config.n_epochs} ... ",
                end="",
                flush=True,
            )
            t_run = time.perf_counter()
            try:
                result = run_one(config, seed)
                all_runs.append(result)
                elapsed = time.perf_counter() - t_run
                warn = f"  [WARN: {result.warning}]" if result.warning else ""
                print(
                    f"eval {result.eval_before:.3e} -> {result.eval_after:.3e}  "
                    f"({result.median_step_time_ms:.1f} ms/step, {elapsed:.1f}s){warn}"
                )
            except Exception as exc:
                elapsed = time.perf_counter() - t_run
                print(f"FAIL after {elapsed:.1f}s -- {type(exc).__name__}: {exc}")
            gc.collect()
            write_partial(all_runs, status="partial")

    print()
    print("== Per-size summary (GATv2) ==")
    summaries = summarise(all_runs)
    print(f"  {'size':<11s} {'n_params':>9s} {'final-eval (med)':>18s} {'best-eval (med)':>17s} {'step (ms)':>11s}")
    print("  " + "-" * 70)
    for s in summaries:
        print(
            f"  {s.size:<11s} {s.n_params:>9d} {s.eval_after_median:>18.3e} "
            f"{s.best_eval_median:>17.3e} {s.median_step_time_ms:>11.1f}"
        )

    out_path = write_partial(all_runs, status="complete")
    overall = time.perf_counter() - t0_overall
    print(f"\nResults written to {out_path}")
    print(f"Total elapsed: {overall:.1f}s ({overall/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

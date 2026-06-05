# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
LinearSystem baseline experimentation.

Trains each of the five built-in ready-to-use GNN sizes (Tiny / Small /
Medium / Large / ExtraLarge), all using LocalSumMessagePassingFunction, on
LinearSystemProblemLoader (DC-power-flow toy). Multiple seeds per size to
capture variance. Reports per-size baseline numbers (final val MSE, step
time, peak memory, convergence behaviour) that subsequent attention work
will be compared against.

Output: benchmarks/results/baseline_linearsystem.json.
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

from energnn.model.ready_to_use import (
    ExtraLargeRecurrentEquivariantGNN,
    LargeRecurrentEquivariantGNN,
    MediumRecurrentEquivariantGNN,
    SmallRecurrentEquivariantGNN,
    TinyRecurrentEquivariantGNN,
)
from energnn.problem.example import LinearSystemProblemLoader
from energnn.trainer import Trainer

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "results" / HERE.name

# Tutorial defaults (tutorial_notebook.ipynb cell 43) extended for stable measurement.
N_MAX = 3
DATASET_SIZE = 64
BATCH_SIZE = 4
VAL_DATASET_SIZE = 32
SEEDS = (0, 1, 2)

# Per-size (n_seeds, n_epochs) budgets. The default scope is the three
# smaller sizes -- this is the configuration intended for baseline
# documentation and for the per-PR pre-commit verification gate. Large and
# ExtraLarge are *available* via the LARGE_SIZES extension below but are
# left out of the default because they add hours of wall time for marginal
# baseline signal on the n_max=3 toy. Enable them by hand when measuring
# perf on the actual production graph sizes.
SIZE_CONFIGS = (
    ("Tiny", TinyRecurrentEquivariantGNN, 3, 10),
    ("Small", SmallRecurrentEquivariantGNN, 3, 15),
)

# Larger sizes are kept here for explicit opt-in; concat into SIZE_CONFIGS
# manually when needed (e.g. ahead of a Gate-6 perf check on production
# graphs).
LARGER_SIZES = (
    ("Medium", MediumRecurrentEquivariantGNN, 3, 20),
    ("Large", LargeRecurrentEquivariantGNN, 3, 25),
    ("ExtraLarge", ExtraLargeRecurrentEquivariantGNN, 3, 30),
)


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
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "flax": _flax.__version__,
        "optax": _optax.__version__,
    }
    return info


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


def measure_step_times(trainer, train_loader, n_steps: int = 50) -> tuple[float, float]:
    """Median and p90 step time in ms, post-warmup."""
    # Warmup
    for problem_batch in train_loader:
        _ = trainer.training_step(problem_batch, get_info=False)
        break
    step_times = []
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


def run_one(size_name: str, size_cls, n_epochs: int, seed: int) -> RunResult:
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

    model = size_cls(
        in_structure=train_loader.context_structure,
        out_structure=train_loader.decision_structure,
        seed=seed,
    )
    n_params = count_params(model)

    trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-3))

    eval_before = eval_score(trainer, val_loader)
    if not np.isfinite(eval_before):
        warning = f"non-finite eval_before={eval_before}"

    median_ms, p90_ms = measure_step_times(trainer, train_loader, n_steps=30)

    epoch_eval_curve: list[float] = []
    t0 = time.perf_counter()
    for _ in range(n_epochs):
        trainer.train(
            train_loader=train_loader,
            val_loader=None,
            n_epochs=1,
            progress_bar=False,
            eval_before_training=False,
            eval_after_epoch=False,
        )
        e = eval_score(trainer, val_loader)
        epoch_eval_curve.append(e)
    total_time_s = time.perf_counter() - t0

    eval_after = epoch_eval_curve[-1] if epoch_eval_curve else eval_before
    eval_improvement = (eval_before - eval_after) if np.isfinite(eval_before) and np.isfinite(eval_after) else float("nan")

    if not np.isfinite(eval_after):
        warning = (warning + "; " if warning else "") + f"non-finite eval_after={eval_after}"

    peak_mb = peak_memory_mb()

    return RunResult(
        size=size_name,
        seed=seed,
        n_epochs=n_epochs,
        n_params=n_params,
        eval_before=eval_before,
        eval_after=eval_after,
        eval_improvement=eval_improvement,
        epoch_eval_curve=epoch_eval_curve,
        median_step_time_ms=median_ms,
        p90_step_time_ms=p90_ms,
        total_train_time_s=total_time_s,
        peak_memory_mb=peak_mb,
        warning=warning,
    )


def summarise(results_by_size: dict[str, list[RunResult]]) -> list[SizeSummary]:
    summaries = []
    for size_name, runs in results_by_size.items():
        finals = [r.eval_after for r in runs if np.isfinite(r.eval_after)]
        if not finals:
            finals = [float("nan")]
        improvements = [r.eval_improvement for r in runs if np.isfinite(r.eval_improvement)]
        median_imp = statistics.median(improvements) if improvements else float("nan")
        median_step = statistics.median(r.median_step_time_ms for r in runs)
        median_mem = statistics.median(r.peak_memory_mb for r in runs)
        summaries.append(
            SizeSummary(
                size=size_name,
                n_seeds=len(runs),
                n_params=runs[0].n_params if runs else 0,
                eval_after_min=min(finals),
                eval_after_median=statistics.median(finals),
                eval_after_max=max(finals),
                eval_improvement_median=median_imp,
                median_step_time_ms=median_step,
                peak_memory_mb_median=median_mem,
            )
        )
    return summaries


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== LinearSystem baseline ==")
    print(f"  n_max={N_MAX}, train_size={DATASET_SIZE}, batch={BATCH_SIZE}, val_size={VAL_DATASET_SIZE}")
    print(f"  seeds={SEEDS}")
    print(f"  sizes={[s for s, _, _, _ in SIZE_CONFIGS]}")
    print()

    results_by_size: dict[str, list[RunResult]] = {}
    all_runs: list[RunResult] = []
    out_path = RESULTS_DIR / "baseline_linearsystem.json"
    for size_name, size_cls, n_seeds, n_epochs in SIZE_CONFIGS:
        results_by_size[size_name] = []
        for seed in SEEDS[:n_seeds]:
            print(f"  {size_name:11s} seed={seed} n_epochs={n_epochs} ... ", end="", flush=True)
            t0 = time.perf_counter()
            result = run_one(size_name, size_cls, n_epochs, seed)
            elapsed = time.perf_counter() - t0
            results_by_size[size_name].append(result)
            all_runs.append(result)
            warn_tag = f" [WARN: {result.warning}]" if result.warning else ""
            print(
                f"eval {result.eval_before:.3e} -> {result.eval_after:.3e} "
                f"({result.median_step_time_ms:.1f} ms/step, {elapsed:.1f}s){warn_tag}"
            )
            gc.collect()
            # Incremental dump so that partial results survive an early stop.
            partial = BenchmarkReport(
                env=env_fingerprint(),
                config={
                    "n_max": N_MAX,
                    "dataset_size": DATASET_SIZE,
                    "batch_size": BATCH_SIZE,
                    "val_dataset_size": VAL_DATASET_SIZE,
                    "seeds": list(SEEDS),
                    "sizes": [{"name": n, "n_seeds": s, "n_epochs": e} for n, _, s, e in SIZE_CONFIGS],
                    "baseline_message_function": "LocalSumMessagePassingFunction",
                    "optimizer": "optax.adam(1e-3)",
                    "status": "partial",
                },
                runs=[asdict(r) for r in all_runs],
                summaries=[asdict(s) for s in summarise(results_by_size)],
            )
            out_path.write_text(json.dumps(asdict(partial), indent=2, default=float))

    summaries = summarise(results_by_size)
    print()
    print("== Per-size summary ==")
    print(f"  {'size':12s} {'n_params':>10s} {'eval min':>11s} {'eval med':>11s} {'eval max':>11s} {'step ms':>10s} {'mem MB':>10s}")
    for s in summaries:
        print(
            f"  {s.size:12s} {s.n_params:>10d} {s.eval_after_min:>11.3e} {s.eval_after_median:>11.3e} "
            f"{s.eval_after_max:>11.3e} {s.median_step_time_ms:>10.1f} {s.peak_memory_mb_median:>10.1f}"
        )

    report = BenchmarkReport(
        env=env_fingerprint(),
        config={
            "n_max": N_MAX,
            "dataset_size": DATASET_SIZE,
            "batch_size": BATCH_SIZE,
            "val_dataset_size": VAL_DATASET_SIZE,
            "seeds": list(SEEDS),
            "sizes": [{"name": n, "n_seeds": s, "n_epochs": e} for n, _, s, e in SIZE_CONFIGS],
            "baseline_message_function": "LocalSumMessagePassingFunction",
            "optimizer": "optax.adam(1e-3)",
            "status": "complete",
        },
        runs=[asdict(r) for r in all_runs],
        summaries=[asdict(s) for s in summaries],
    )
    out_path.write_text(json.dumps(asdict(report), indent=2, default=float))
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

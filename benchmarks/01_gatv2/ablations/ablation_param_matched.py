# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Ablation: param-matched GATv2 vs LocalSum.

Hypothesis: the small improvements observed for GATv2 over LocalSum on
the supervised IEEE benchmark could be attributable simply to GATv2's
higher parameter count (+45 percent at Small) rather than to the
attention mechanism itself. This ablation tests that hypothesis by
running GATv2 with reduced hidden width (hidden_sizes=[10] vs the
default [16]) so the resulting model has approximately the same number
of parameters as LocalSum Small (~14495 vs 15863). If param-matched
GATv2 still beats LocalSum, the attention mechanism contributes
genuine value; if it ties or loses, the earlier improvements were attributable
to capacity, not mechanism.

Tests on ieee9 / ieee14 / ieee30 with Small-equivalent base config,
3 seeds, 15 epochs.

Output JSON consumed by render_baselines_md.py.
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
from energnn.trainer import Trainer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
from ac_load_flow_problem import ACLoadFlowProblemLoader

RESULTS_DIR = HERE.parent.parent / "results" / HERE.parent.name

IEEE_SIZES = (9, 14, 30)
PERTURBATION_SCALE = 0.1
DATASET_SIZE = 32
BATCH_SIZE = 4
VAL_DATASET_SIZE = 16
SEEDS = (0, 1, 2)
N_EPOCHS = 15

# Param-matched GATv2 config: hidden_sizes=[10] gives ~14495 params,
# within ~9 percent of the LocalSum Small baseline (15863 params), the
# nearest below the target. Hidden=[12] would land ~9 percent above.
# Keeping just below the LocalSum baseline favours the null hypothesis
# (so if param-matched GATv2 still outperforms LocalSum, the improvement
# is attributable to mechanism, not capacity).
LATENT_DIM = 8
HIDDEN_SIZES_PARAM_MATCHED: tuple[int, ...] = (10,)
N_STEPS = 10
N_BREAKPOINTS = 20


def env_fingerprint() -> dict:
    import flax as _flax
    import pypowsybl as _pypsbl
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "flax": _flax.__version__,
        "pypowsybl": _pypsbl.__version__,
    }


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024  # ru_maxrss is kilobytes on Linux


def count_params(model) -> int:
    _, params, _ = nnx.split(model, nnx.Param, ...)
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params) if hasattr(leaf, "size"))


def build_param_matched_gatv2(in_structure, out_structure, *, seed: int) -> GNN:
    rngs = nnx.Rngs(seed)
    normalizer = TDigestNormalizer(in_structure=in_structure, n_breakpoints=N_BREAKPOINTS, update_limit=1000)
    encoder = MLPEncoder(
        in_structure=in_structure, hidden_sizes=list(HIDDEN_SIZES_PARAM_MATCHED),
        activation=nnx.leaky_relu, out_size=LATENT_DIM,
        use_bias=True, final_activation=None, rngs=rngs,
    )
    msg_fn = GATv2MessagePassingFunction(
        in_graph_structure=in_structure, in_array_size=LATENT_DIM,
        hidden_sizes=list(HIDDEN_SIZES_PARAM_MATCHED),
        activation=nnx.leaky_relu, out_size=LATENT_DIM,
        use_bias=True, final_activation=None,
        outer_activation=nnx.tanh, encoded_feature_size=LATENT_DIM, rngs=rngs,
    )
    phi = MLP(
        in_size=LATENT_DIM, hidden_sizes=[], activation=nnx.leaky_relu,
        out_size=LATENT_DIM, use_bias=True, final_activation=nnx.tanh, rngs=rngs,
    )
    coupler = RecurrentCoupler(phi=phi, message_functions=[msg_fn], n_steps=N_STEPS)
    decoder = MLPEquivariantDecoder(
        in_graph_structure=in_structure, in_array_size=LATENT_DIM,
        hidden_sizes=list(HIDDEN_SIZES_PARAM_MATCHED),
        activation=nnx.leaky_relu, out_structure=out_structure,
        use_bias=True, final_activation=None, encoded_feature_size=LATENT_DIM, rngs=rngs,
    )
    return GNN(normalizer=normalizer, encoder=encoder, coupler=coupler, decoder=decoder)


@dataclass
class RunResult:
    network: str
    seed: int
    n_params: int
    n_epochs: int
    eval_before: float
    eval_after: float
    epoch_eval_curve: list[float]
    total_train_time_s: float


@dataclass
class Summary:
    network: str
    n_seeds: int
    n_params: int
    eval_after_min: float
    eval_after_median: float
    eval_after_max: float
    best_eval_median: float


@dataclass
class Report:
    env: dict
    config: dict
    runs: list[dict] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)


def run_one(network: str, seed: int) -> RunResult:
    train_loader = ACLoadFlowProblemLoader(
        network_name=network, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE,
        seed=10 * seed + 7, perturbation_scale=PERTURBATION_SCALE,
    )
    val_loader = ACLoadFlowProblemLoader(
        network_name=network, dataset_size=VAL_DATASET_SIZE, batch_size=BATCH_SIZE,
        seed=10 * seed + 8, perturbation_scale=PERTURBATION_SCALE,
    )
    model = build_param_matched_gatv2(
        in_structure=train_loader.context_structure,
        out_structure=train_loader.decision_structure,
        seed=seed,
    )
    n_params = count_params(model)
    trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-3))

    eval_before, _ = trainer.eval(val_loader, progress_bar=False)
    eval_before = float(eval_before)

    epoch_eval_curve: list[float] = []
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS):
        trainer.train(
            train_loader=train_loader, val_loader=None, n_epochs=1,
            progress_bar=False, eval_before_training=False, eval_after_epoch=False,
        )
        s, _ = trainer.eval(val_loader, progress_bar=False)
        epoch_eval_curve.append(float(s))
    total_time = time.perf_counter() - t0

    return RunResult(
        network=network, seed=seed, n_params=n_params, n_epochs=N_EPOCHS,
        eval_before=eval_before, eval_after=epoch_eval_curve[-1],
        epoch_eval_curve=epoch_eval_curve, total_train_time_s=total_time,
    )


def summarise(runs: list[RunResult]) -> list[Summary]:
    by_net: dict[str, list[RunResult]] = {}
    for r in runs:
        by_net.setdefault(r.network, []).append(r)
    out = []
    for net, rs in by_net.items():
        finals = [r.eval_after for r in rs if np.isfinite(r.eval_after)] or [float("nan")]
        bests = []
        for r in rs:
            curve = [v for v in r.epoch_eval_curve if np.isfinite(v)]
            if curve:
                bests.append(min(curve))
        if not bests:
            bests = [float("nan")]
        out.append(
            Summary(
                network=net, n_seeds=len(rs), n_params=rs[0].n_params,
                eval_after_min=min(finals),
                eval_after_median=statistics.median(finals),
                eval_after_max=max(finals),
                best_eval_median=statistics.median(bests),
            )
        )
    return out


def write_report(all_runs: list[RunResult], status: str) -> Path:
    report = Report(
        env=env_fingerprint(),
        config={
            "ieee_sizes": list(IEEE_SIZES),
            "perturbation_scale": PERTURBATION_SCALE,
            "dataset_size": DATASET_SIZE,
            "batch_size": BATCH_SIZE,
            "val_dataset_size": VAL_DATASET_SIZE,
            "seeds": list(SEEDS),
            "n_epochs": N_EPOCHS,
            "latent_dim": LATENT_DIM,
            "hidden_sizes_param_matched": list(HIDDEN_SIZES_PARAM_MATCHED),
            "message_function": "GATv2MessagePassingFunction (param-matched: hidden=[10])",
            "optimizer": "optax.adam(1e-3)",
            "status": status,
            "compared_against": "baseline_ac_load_flow.json (LocalSum Small, 15863 params)",
            "hypothesis": (
                "If param-matched GATv2 (~14495 params, slightly below LocalSum's "
                "15863) still matches or beats LocalSum, the attention mechanism "
                "contributes value independent of capacity. If it ties or loses, "
                "the earlier GATv2 improvements were attributable to its extra parameters."
            ),
        },
        runs=[asdict(r) for r in all_runs],
        summaries=[asdict(s) for s in summarise(all_runs)],
    )
    out_path = RESULTS_DIR / "ablation_param_matched.json"
    out_path.write_text(json.dumps(asdict(report), indent=2, default=float))
    return out_path


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== Ablation: param-matched GATv2 (hidden=[10]) vs LocalSum baseline ==")
    print(f"  ieee_sizes: {IEEE_SIZES}, seeds: {SEEDS}, n_epochs: {N_EPOCHS}")
    print()

    all_runs: list[RunResult] = []
    total = len(IEEE_SIZES) * len(SEEDS)
    idx = 0
    t0_overall = time.perf_counter()
    for size_int in IEEE_SIZES:
        network = f"ieee{size_int}"
        for seed in SEEDS:
            idx += 1
            print(f"  [{idx:2d}/{total}] {network:<6s} seed={seed} ... ", end="", flush=True)
            t_run = time.perf_counter()
            try:
                result = run_one(network, seed)
                all_runs.append(result)
                elapsed = time.perf_counter() - t_run
                print(f"eval {result.eval_before:.3e} -> {result.eval_after:.3e}  ({elapsed:.1f}s)")
            except Exception as exc:
                elapsed = time.perf_counter() - t_run
                print(f"FAIL after {elapsed:.1f}s -- {type(exc).__name__}: {exc}")
            gc.collect()
            write_report(all_runs, status="partial")

    print()
    print("== Per-network summary ==")
    summaries = summarise(all_runs)
    print(f"  {'network':<7s} {'n_params':>9s} {'eval_med':>11s} {'best_eval_med':>14s}")
    print("  " + "-" * 50)
    for s in summaries:
        print(
            f"  {s.network:<7s} {s.n_params:>9d} {s.eval_after_median:>11.3e} {s.best_eval_median:>14.3e}"
        )

    out_path = write_report(all_runs, status="complete")
    overall = time.perf_counter() - t0_overall
    print(f"\nResults written to {out_path}")
    print(f"Total elapsed: {overall:.1f}s ({overall/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

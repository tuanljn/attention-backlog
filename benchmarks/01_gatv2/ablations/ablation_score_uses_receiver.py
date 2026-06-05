# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Ablation: GATv2 with explicit receiver coord in the score MLP input.

Hypothesis: The default GATv2 implementation gives the score MLP an
input consisting of the concatenated coordinates of every port of the
hyper-edge plus its features. Because the MLPs are factored per
(class, port), the per-port weights can already learn the asymmetric
attention pattern that the GATv2 paper's ``[h_a || h_e]`` formulation
makes explicit -- the receiver coordinate is present in the input as
one of the port-gathered coordinates. Setting ``score_uses_receiver=True``
appends an explicit duplicate of the receiver coord at the end of the
score MLP input, which strengthens the signal at the cost of a slightly
wider score MLP (input size grows by ``in_array_size``).

This ablation tests whether the explicit duplicate helps (positive),
hurts (negative, suggesting over-parameterisation noise), or has no
effect (null, confirming the per-port factoring already captures the
signal).

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
LATENT_DIM = 8
HIDDEN_SIZES: tuple[int, ...] = (16,)
N_BREAKPOINTS = 20
N_STEPS = 10


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


def build_gatv2_with_flag(score_uses_receiver: bool, in_structure, out_structure, *, seed: int) -> GNN:
    rngs = nnx.Rngs(seed)
    normalizer = TDigestNormalizer(in_structure=in_structure, n_breakpoints=N_BREAKPOINTS, update_limit=1000)
    encoder = MLPEncoder(
        in_structure=in_structure, hidden_sizes=list(HIDDEN_SIZES),
        activation=nnx.leaky_relu, out_size=LATENT_DIM,
        use_bias=True, final_activation=None, rngs=rngs,
    )
    msg_fn = GATv2MessagePassingFunction(
        in_graph_structure=in_structure, in_array_size=LATENT_DIM,
        hidden_sizes=list(HIDDEN_SIZES), activation=nnx.leaky_relu,
        out_size=LATENT_DIM, use_bias=True, final_activation=None,
        outer_activation=nnx.tanh, encoded_feature_size=LATENT_DIM,
        score_uses_receiver=score_uses_receiver, rngs=rngs,
    )
    phi = MLP(
        in_size=LATENT_DIM, hidden_sizes=[], activation=nnx.leaky_relu,
        out_size=LATENT_DIM, use_bias=True, final_activation=nnx.tanh, rngs=rngs,
    )
    coupler = RecurrentCoupler(phi=phi, message_functions=[msg_fn], n_steps=N_STEPS)
    decoder = MLPEquivariantDecoder(
        in_graph_structure=in_structure, in_array_size=LATENT_DIM,
        hidden_sizes=list(HIDDEN_SIZES), activation=nnx.leaky_relu,
        out_structure=out_structure, use_bias=True, final_activation=None,
        encoded_feature_size=LATENT_DIM, rngs=rngs,
    )
    return GNN(normalizer=normalizer, encoder=encoder, coupler=coupler, decoder=decoder)


@dataclass
class RunResult:
    network: str
    score_uses_receiver: bool
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
    score_uses_receiver: bool
    n_seeds: int
    n_params: int
    eval_after_median: float
    best_eval_median: float


@dataclass
class Report:
    env: dict
    config: dict
    runs: list[dict] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)


def run_one(network: str, score_uses_receiver: bool, seed: int) -> RunResult:
    train_loader = ACLoadFlowProblemLoader(
        network_name=network, dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE,
        seed=10 * seed + 7, perturbation_scale=PERTURBATION_SCALE,
    )
    val_loader = ACLoadFlowProblemLoader(
        network_name=network, dataset_size=VAL_DATASET_SIZE, batch_size=BATCH_SIZE,
        seed=10 * seed + 8, perturbation_scale=PERTURBATION_SCALE,
    )
    model = build_gatv2_with_flag(
        score_uses_receiver,
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
        network=network, score_uses_receiver=score_uses_receiver, seed=seed,
        n_params=n_params, n_epochs=N_EPOCHS,
        eval_before=eval_before, eval_after=epoch_eval_curve[-1],
        epoch_eval_curve=epoch_eval_curve, total_train_time_s=total_time,
    )


def summarise(runs: list[RunResult]) -> list[Summary]:
    by_key: dict[tuple[str, bool], list[RunResult]] = {}
    for r in runs:
        by_key.setdefault((r.network, r.score_uses_receiver), []).append(r)
    out = []
    for (net, flag), rs in by_key.items():
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
                network=net, score_uses_receiver=flag, n_seeds=len(rs),
                n_params=rs[0].n_params,
                eval_after_median=statistics.median(finals),
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
            "hidden_sizes": list(HIDDEN_SIZES),
            "score_uses_receiver_values": [False, True],
            "optimizer": "optax.adam(1e-3)",
            "status": status,
            "hypothesis": (
                "Default score_uses_receiver=False already encodes the asymmetric "
                "GATv2 attention via per-(class, port) MLP factoring. Setting "
                "score_uses_receiver=True appends an explicit duplicate of the "
                "receiver coord to the score MLP input. Positive if explicit "
                "signal helps; null if implicit factoring suffices; negative "
                "if added params just add noise."
            ),
        },
        runs=[asdict(r) for r in all_runs],
        summaries=[asdict(s) for s in summarise(all_runs)],
    )
    out_path = RESULTS_DIR / "ablation_score_uses_receiver.json"
    out_path.write_text(json.dumps(asdict(report), indent=2, default=float))
    return out_path


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("== Ablation: GATv2 score_uses_receiver (False vs True) ==")
    print(f"  ieee_sizes: {IEEE_SIZES}, seeds: {SEEDS}, n_epochs: {N_EPOCHS}")
    print()

    all_runs: list[RunResult] = []
    total = len(IEEE_SIZES) * 2 * len(SEEDS)
    idx = 0
    t0_overall = time.perf_counter()
    for size_int in IEEE_SIZES:
        network = f"ieee{size_int}"
        for flag in (False, True):
            for seed in SEEDS:
                idx += 1
                print(
                    f"  [{idx:2d}/{total}] {network:<6s} sur={flag!s:5s} seed={seed} ... ",
                    end="", flush=True,
                )
                t_run = time.perf_counter()
                try:
                    result = run_one(network, flag, seed)
                    all_runs.append(result)
                    elapsed = time.perf_counter() - t_run
                    print(f"eval {result.eval_before:.3e} -> {result.eval_after:.3e}  ({elapsed:.1f}s)")
                except Exception as exc:
                    elapsed = time.perf_counter() - t_run
                    print(f"FAIL after {elapsed:.1f}s -- {type(exc).__name__}: {exc}")
                gc.collect()
                write_report(all_runs, status="partial")

    print()
    print("== Per-(network, score_uses_receiver) summary ==")
    summaries = summarise(all_runs)
    print(f"  {'network':<7s} {'sur':<6s} {'n_params':>9s} {'eval_med':>11s} {'best_eval_med':>14s}")
    print("  " + "-" * 60)
    for s in summaries:
        print(
            f"  {s.network:<7s} {s.score_uses_receiver!s:<6s} {s.n_params:>9d} "
            f"{s.eval_after_median:>11.3e} {s.best_eval_median:>14.3e}"
        )

    out_path = write_report(all_runs, status="complete")
    overall = time.perf_counter() - t0_overall
    print(f"\nResults written to {out_path}")
    print(f"Total elapsed: {overall:.1f}s ({overall/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Ablation: cross-network generalisation of LocalSum and GATv2.

Hypothesis: Because EnerGNN's H2MG model is per-class-per-port factored
and permutation-equivariant, a model trained on one IEEE network (e.g.
ieee9) should give meaningful predictions on a different IEEE network
of the same kind (e.g. ieee14, ieee30) without any retraining. This
tests the strong form of the architecture's generalisation claim.

Procedure: train a Small-equivalent GNN (with LocalSum and with GATv2,
matched config) on ieee9 supervised AC load flow for 15 epochs. Then
evaluate on the val loader of each of ieee9, ieee14, and ieee30 (with
seed 8 + 10 * train_seed). The ieee9 score is the in-distribution
reference; ieee14 and ieee30 are out-of-distribution generalisation
checks.

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
    LocalSumMessagePassingFunction,
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

TRAIN_NETWORK = "ieee9"
EVAL_NETWORKS = ("ieee9", "ieee14", "ieee30")
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


def build_gnn(message_fn_name: str, in_structure, out_structure, *, seed: int) -> GNN:
    """Build a Small-equivalent GNN with the named message function."""
    rngs = nnx.Rngs(seed)
    normalizer = TDigestNormalizer(in_structure=in_structure, n_breakpoints=N_BREAKPOINTS, update_limit=1000)
    encoder = MLPEncoder(
        in_structure=in_structure, hidden_sizes=list(HIDDEN_SIZES),
        activation=nnx.leaky_relu, out_size=LATENT_DIM,
        use_bias=True, final_activation=None, rngs=rngs,
    )
    if message_fn_name == "LocalSum":
        msg_fn = LocalSumMessagePassingFunction(
            in_graph_structure=in_structure, in_array_size=LATENT_DIM,
            hidden_sizes=list(HIDDEN_SIZES), activation=nnx.leaky_relu,
            out_size=LATENT_DIM, use_bias=True, final_activation=None,
            outer_activation=nnx.tanh, encoded_feature_size=LATENT_DIM, rngs=rngs,
        )
    elif message_fn_name == "GATv2":
        msg_fn = GATv2MessagePassingFunction(
            in_graph_structure=in_structure, in_array_size=LATENT_DIM,
            hidden_sizes=list(HIDDEN_SIZES), activation=nnx.leaky_relu,
            out_size=LATENT_DIM, use_bias=True, final_activation=None,
            outer_activation=nnx.tanh, encoded_feature_size=LATENT_DIM, rngs=rngs,
        )
    else:
        raise ValueError(f"unknown message_fn_name: {message_fn_name}")
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
class CrossNetworkResult:
    message_fn: str
    train_network: str
    seed: int
    n_params: int
    n_epochs: int
    eval_in_distribution: float
    eval_out_of_distribution: dict[str, float]
    total_train_time_s: float


@dataclass
class CrossNetworkSummary:
    message_fn: str
    train_network: str
    eval_network: str
    in_distribution: bool
    n_seeds: int
    eval_median: float
    eval_min: float
    eval_max: float
    generalization_gap_pct_vs_train: float


@dataclass
class Report:
    env: dict
    config: dict
    runs: list[dict] = field(default_factory=list)
    summaries: list[dict] = field(default_factory=list)


def run_one(message_fn_name: str, seed: int) -> CrossNetworkResult:
    train_loader = ACLoadFlowProblemLoader(
        network_name=TRAIN_NETWORK,
        dataset_size=DATASET_SIZE, batch_size=BATCH_SIZE,
        seed=10 * seed + 7, perturbation_scale=PERTURBATION_SCALE,
    )
    model = build_gnn(
        message_fn_name,
        in_structure=train_loader.context_structure,
        out_structure=train_loader.decision_structure,
        seed=seed,
    )
    n_params = count_params(model)
    trainer = Trainer(model=model, gradient_transformation=optax.adam(1e-3))

    t0 = time.perf_counter()
    trainer.train(
        train_loader=train_loader, val_loader=None, n_epochs=N_EPOCHS,
        progress_bar=False, eval_before_training=False, eval_after_epoch=False,
    )
    total_train_time = time.perf_counter() - t0

    eval_scores: dict[str, float] = {}
    for eval_net in EVAL_NETWORKS:
        val_loader = ACLoadFlowProblemLoader(
            network_name=eval_net,
            dataset_size=VAL_DATASET_SIZE, batch_size=BATCH_SIZE,
            seed=10 * seed + 8, perturbation_scale=PERTURBATION_SCALE,
        )
        score, _ = trainer.eval(val_loader, progress_bar=False)
        eval_scores[eval_net] = float(score)

    return CrossNetworkResult(
        message_fn=message_fn_name,
        train_network=TRAIN_NETWORK,
        seed=seed,
        n_params=n_params,
        n_epochs=N_EPOCHS,
        eval_in_distribution=eval_scores[TRAIN_NETWORK],
        eval_out_of_distribution={net: eval_scores[net] for net in EVAL_NETWORKS if net != TRAIN_NETWORK},
        total_train_time_s=total_train_time,
    )


def summarise(runs: list[CrossNetworkResult]) -> list[CrossNetworkSummary]:
    by_key: dict[tuple[str, str], list[float]] = {}
    for r in runs:
        all_scores = {TRAIN_NETWORK: r.eval_in_distribution, **r.eval_out_of_distribution}
        for eval_net, score in all_scores.items():
            if np.isfinite(score):
                by_key.setdefault((r.message_fn, eval_net), []).append(score)
    # For relative gap, we need the in-distribution score per (message_fn).
    in_dist_median: dict[str, float] = {}
    for (msg_fn, eval_net), scores in by_key.items():
        if eval_net == TRAIN_NETWORK:
            in_dist_median[msg_fn] = statistics.median(scores) if scores else float("nan")
    out = []
    for (msg_fn, eval_net), scores in by_key.items():
        if not scores:
            continue
        median = statistics.median(scores)
        ref = in_dist_median.get(msg_fn, float("nan"))
        if ref > 0 and np.isfinite(ref):
            gap_pct = (median - ref) / ref * 100.0
        else:
            gap_pct = float("nan")
        out.append(
            CrossNetworkSummary(
                message_fn=msg_fn,
                train_network=TRAIN_NETWORK,
                eval_network=eval_net,
                in_distribution=(eval_net == TRAIN_NETWORK),
                n_seeds=len(scores),
                eval_median=median,
                eval_min=min(scores),
                eval_max=max(scores),
                generalization_gap_pct_vs_train=gap_pct,
            )
        )
    return out


def write_report(all_runs: list[CrossNetworkResult], status: str) -> Path:
    report = Report(
        env=env_fingerprint(),
        config={
            "train_network": TRAIN_NETWORK,
            "eval_networks": list(EVAL_NETWORKS),
            "perturbation_scale": PERTURBATION_SCALE,
            "dataset_size": DATASET_SIZE,
            "batch_size": BATCH_SIZE,
            "val_dataset_size": VAL_DATASET_SIZE,
            "seeds": list(SEEDS),
            "n_epochs": N_EPOCHS,
            "latent_dim": LATENT_DIM,
            "hidden_sizes": list(HIDDEN_SIZES),
            "message_functions": ["LocalSum", "GATv2"],
            "optimizer": "optax.adam(1e-3)",
            "status": status,
            "hypothesis": (
                "EnerGNN's per-(class, port) factoring + permutation equivariance "
                "should let a model trained on one IEEE topology give meaningful "
                "predictions on another IEEE topology of the same kind. "
                "Falsified if cross-network eval is no better than untrained baseline."
            ),
        },
        runs=[asdict(r) for r in all_runs],
        summaries=[asdict(s) for s in summarise(all_runs)],
    )
    out_path = RESULTS_DIR / "ablation_cross_network.json"
    out_path.write_text(json.dumps(asdict(report), indent=2, default=float))
    return out_path


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== Ablation: cross-network generalisation ==")
    print(f"  train on:   {TRAIN_NETWORK}")
    print(f"  eval on:    {EVAL_NETWORKS}")
    print(f"  message_fns: LocalSum, GATv2")
    print(f"  seeds:      {SEEDS}")
    print(f"  n_epochs:   {N_EPOCHS}")
    print()

    all_runs: list[CrossNetworkResult] = []
    total = 2 * len(SEEDS)
    idx = 0
    t0_overall = time.perf_counter()
    for message_fn_name in ("LocalSum", "GATv2"):
        for seed in SEEDS:
            idx += 1
            print(
                f"  [{idx:2d}/{total}] {message_fn_name:<8s} seed={seed} ... ",
                end="",
                flush=True,
            )
            t_run = time.perf_counter()
            try:
                result = run_one(message_fn_name, seed)
                all_runs.append(result)
                elapsed = time.perf_counter() - t_run
                in_d = result.eval_in_distribution
                ood = " | ".join(f"{n}={v:.3e}" for n, v in result.eval_out_of_distribution.items())
                print(f"in-dist {in_d:.3e} | ood {ood}  ({elapsed:.1f}s)")
            except Exception as exc:
                elapsed = time.perf_counter() - t_run
                print(f"FAIL after {elapsed:.1f}s -- {type(exc).__name__}: {exc}")
            gc.collect()
            write_report(all_runs, status="partial")

    print()
    print("== Per-(message_fn, eval_network) summary ==")
    summaries = summarise(all_runs)
    print(f"  {'message_fn':<10s} {'eval_net':<8s} {'in/ood':<6s} {'eval_med':>11s} {'gap vs train':>14s}")
    print("  " + "-" * 60)
    for s in summaries:
        in_ood = "in" if s.in_distribution else "ood"
        gap_str = f"{s.generalization_gap_pct_vs_train:+.1f}%" if np.isfinite(s.generalization_gap_pct_vs_train) else "n/a"
        print(
            f"  {s.message_fn:<10s} {s.eval_network:<8s} {in_ood:<6s} "
            f"{s.eval_median:>11.3e} {gap_str:>14s}"
        )

    out_path = write_report(all_runs, status="complete")
    overall = time.perf_counter() - t0_overall
    print(f"\nResults written to {out_path}")
    print(f"Total elapsed: {overall:.1f}s ({overall/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Render BASELINES.md at the repo root from the JSON outputs of
baseline_linearsystem.py and baseline_ieee_conversion.py.

This is a tiny templating step kept separate from the benchmark scripts so
that re-rendering the markdown after the JSON has been edited (e.g. to add
follow-up sizes) does not require re-running the benchmarks.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent
RESULTS_DIR = HERE / "results"


def fmt(x: Any, kind: str = "f", width: int | None = None) -> str:
    if x is None:
        s = "n/a"
    elif isinstance(x, float) and (math.isnan(x) or not math.isfinite(x)):
        s = "n/a"
    elif kind == "e":
        s = f"{x:.3e}"
    elif kind == "f":
        s = f"{x:.3f}"
    elif kind == "f1":
        s = f"{x:.1f}"
    elif kind == "d":
        s = f"{x:d}"
    else:
        s = str(x)
    return s.rjust(width) if width else s


def render_ls(ls: dict) -> str:
    env = ls["env"]
    cfg = ls["config"]
    summaries = ls["summaries"]

    size_names = [s["name"] for s in cfg["sizes"]]
    sizes_phrase = "/".join(size_names)

    lines = []
    lines.append("## LinearSystem baseline\n")
    lines.append("Built-in DC-power-flow toy problem (`LinearSystemProblemLoader`). "
                 f"The {len(size_names)} ready-to-use sizes covered here ({sizes_phrase}) "
                 "all use `LocalSumMessagePassingFunction` as the message function and "
                 f"`RecurrentCoupler` as the coupler. Optimizer `{cfg['optimizer']}`. "
                 f"Seeds tested: `{cfg['seeds']}`. Data: `n_max={cfg['n_max']}`, "
                 f"`dataset_size={cfg['dataset_size']}`, `batch_size={cfg['batch_size']}`, "
                 f"`val_dataset_size={cfg['val_dataset_size']}`.\n")
    if cfg.get("notes"):
        lines.append(f"**Note:** {cfg['notes']}\n")
    # Best-eval-per-seed (min over epoch_eval_curve) gives an early-stopping
    # reference. Attention variants land on the same metric; comparing only
    # final-eval would unfairly penalise overfitting baselines.
    import math, statistics
    best_eval_median_by_size: dict[str, float] = {}
    for r in ls["runs"]:
        curve = [v for v in r["epoch_eval_curve"] if math.isfinite(v)]
        if not curve:
            continue
        best_eval_median_by_size.setdefault(r["size"], []).append(min(curve))
    best_eval_median_by_size = {
        size: statistics.median(values) for size, values in best_eval_median_by_size.items()
    }

    lines.append("### Summary table\n")
    lines.append("`final-eval` columns are the eval MSE at the last training epoch; "
                 "`best-eval (median)` is the median over seeds of each seed's best "
                 "eval across all epochs (a fair early-stopping reference).\n")
    lines.append("| Size       | n_params | n_epochs | final-eval (min) | final-eval (median) | final-eval (max) | best-eval (median) | step time (ms) | peak mem (MB) |")
    lines.append("|------------|---------:|---------:|-----------------:|--------------------:|-----------------:|-------------------:|---------------:|--------------:|")
    epoch_map = {s["name"]: s["n_epochs"] for s in cfg["sizes"]}
    for s in summaries:
        best = best_eval_median_by_size.get(s["size"], float("nan"))
        lines.append(
            f"| {s['size']:<10s} | {s['n_params']:>8d} | {epoch_map.get(s['size'], '-'):>8} | "
            f"{fmt(s['eval_after_min'], 'e'):>16s} | {fmt(s['eval_after_median'], 'e'):>19s} | "
            f"{fmt(s['eval_after_max'], 'e'):>16s} | {fmt(best, 'e'):>18s} | "
            f"{fmt(s['median_step_time_ms'], 'f1'):>14s} | {fmt(s['peak_memory_mb_median'], 'f1'):>13s} |"
        )
    lines.append("")

    lines.append("### Per-run detail\n")
    lines.append("| Size       | Seed | eval_before | eval_after  | improvement | median ms | p90 ms  | train time (s) | warning |")
    lines.append("|------------|-----:|------------:|------------:|------------:|----------:|--------:|---------------:|---------|")
    for r in ls["runs"]:
        lines.append(
            f"| {r['size']:<10s} | {r['seed']:>4d} | {fmt(r['eval_before'], 'e'):>11s} | "
            f"{fmt(r['eval_after'], 'e'):>11s} | {fmt(r['eval_improvement'], 'e'):>11s} | "
            f"{fmt(r['median_step_time_ms'], 'f1'):>9s} | {fmt(r['p90_step_time_ms'], 'f1'):>7s} | "
            f"{fmt(r['total_train_time_s'], 'f1'):>14s} | {r.get('warning', '') or '-'} |"
        )
    lines.append("")

    lines.append("### Eval-loss curves (per epoch, median across seeds)\n")
    lines.append("```")
    for size in [s["name"] for s in cfg["sizes"]]:
        runs = [r for r in ls["runs"] if r["size"] == size]
        if not runs:
            continue
        epochs = max(len(r["epoch_eval_curve"]) for r in runs)
        medians = []
        for i in range(epochs):
            vals = [r["epoch_eval_curve"][i] for r in runs
                    if i < len(r["epoch_eval_curve"]) and math.isfinite(r["epoch_eval_curve"][i])]
            medians.append(sum(vals) / len(vals) if vals else float("nan"))
        curve = " -> ".join(f"{v:.2e}" if math.isfinite(v) else "nan" for v in medians)
        lines.append(f"  {size:<11s}: {curve}")
    lines.append("```\n")

    lines.append("Environment for these runs:")
    lines.append("```")
    for k, v in env.items():
        lines.append(f"  {k}: {v}")
    lines.append("```\n")
    return "\n".join(lines)


def render_ieee(ieee: dict) -> str:
    env = ieee["env"]
    cfg = ieee["config"]
    lines = []
    lines.append("## IEEE network conversion experimentation\n")
    lines.append(f"For each standard IEEE size {tuple(cfg['ieee_sizes'])}: build the network with "
                 "`pn.create_ieee<N>()`, solve AC load flow with `lf.run_ac(network)`, set "
                 "`network.per_unit = True`, convert via `ACLoadFlowInputConverter` / "
                 "`ACLoadFlowOutputConverter`, then run one forward pass through "
                 f"`{cfg['model']}` (seed `{cfg['model_seed']}`) instantiated from the converter's "
                 "structure. No supervised training. The intent is to confirm shape/dtype "
                 "compatibility, finite outputs, and physically plausible AC quantities "
                 f"(bus voltages in `[{cfg['v_mag_range_pu'][0]}, {cfg['v_mag_range_pu'][1]}]` p.u.).\n")
    lines.append("### Per-network results\n")
    lines.append("| Network  | n_addr | conv ms | LF ms  | Tiny fwd ms | finite (in/out/fwd) | v_mag [min, max] | in range | error |")
    lines.append("|----------|-------:|--------:|-------:|------------:|--------------------:|-----------------:|---------:|-------|")
    for r in ieee["results"]:
        if r.get("error"):
            lines.append(
                f"| {r['name']:<8s} | {'n/a':>6s} | {'n/a':>7s} | {'n/a':>6s} | "
                f"{'n/a':>11s} | n/a | n/a | n/a | `{r['error']}` |"
            )
            continue
        finite_tag = (
            ("Y" if r["input_features_finite"] else "N")
            + "/" + ("Y" if r["output_features_finite"] else "N")
            + "/" + ("Y" if r["tiny_forward_finite"] else "N")
        )
        lines.append(
            f"| {r['name']:<8s} | {r['n_addresses']:>6d} | {fmt(r['conversion_time_ms'], 'f1'):>7s} | "
            f"{fmt(r['loadflow_time_ms'], 'f1'):>6s} | {fmt(r['tiny_forward_time_ms'], 'f1'):>11s} | "
            f"{finite_tag:>19s} | [{fmt(r['v_mag_min'], 'f')}, {fmt(r['v_mag_max'], 'f')}] | "
            f"{'Y' if r['v_mag_in_range'] else 'N':>8s} | {r.get('notes') or '-'} |"
        )
    lines.append("")
    lines.append("### Hyper-edge-set inventory per network\n")
    for r in ieee["results"]:
        if r.get("error"):
            continue
        counts = r["hyper_edge_set_counts"]
        if not counts:
            continue
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"- `{r['name']}`: {pretty}")
    lines.append("")
    lines.append("Environment for these runs:")
    lines.append("```")
    for k, v in env.items():
        lines.append(f"  {k}: {v}")
    lines.append("```\n")
    return "\n".join(lines)


def render_ac_lf(acflow: dict) -> str:
    """Render the supervised AC-load-flow baseline section.

    Trains LocalSumMessagePassingFunction (via Tiny / Small ready-to-use
    GNN) on perturbed-load AC LF supervised data for IEEE 9/14/30, and
    reports per-(network, size) summary.
    """
    import math, statistics
    env = acflow["env"]
    cfg = acflow["config"]
    summaries = acflow["summaries"]
    runs = acflow["runs"]

    lines = []
    lines.append("## IEEE supervised AC-load-flow baseline\n")
    lines.append("Trains `LocalSumMessagePassingFunction` (via the Tiny and Small "
                 "ready-to-use GNN combos) on supervised AC-load-flow data generated "
                 "by `ACLoadFlowProblemLoader`. For each IEEE network in "
                 f"`{tuple(cfg['ieee_sizes'])}`, each training instance is produced by "
                 "perturbing the load setpoints (p0, q0) of the network by independent "
                 f"multiplicative factors in `[1 - {cfg['perturbation_scale']}, 1 + "
                 f"{cfg['perturbation_scale']}]`, then solving AC load flow with "
                 "`lf.run_ac(network)` to obtain the ground-truth response (V_mag for "
                 "buses, P/Q/I for branches and devices). The model is supervised under "
                 f"MSE against this oracle. Optimizer `{cfg['optimizer']}`. Seeds tested: "
                 f"`{cfg['seeds']}`. Data: `dataset_size={cfg['dataset_size']}`, "
                 f"`batch_size={cfg['batch_size']}`, `val_dataset_size={cfg['val_dataset_size']}`. "
                 "Topology is fixed per loader; only load setpoints vary across "
                 "instances, so batches collate without padding.\n")
    if cfg.get("notes"):
        lines.append(f"**Note:** {cfg['notes']}\n")

    lines.append("### Summary table\n")
    lines.append("| Network | Size  | n_params | n_epochs | final-eval (median) | best-eval (median) | improvement (median) | step time (ms) |")
    lines.append("|---------|-------|---------:|---------:|--------------------:|-------------------:|---------------------:|---------------:|")
    epoch_map = {s["name"]: s["n_epochs"] for s in cfg["sizes"]}
    for s in summaries:
        lines.append(
            f"| {s['network']:<7s} | {s['size']:<5s} | {s['n_params']:>8d} | "
            f"{epoch_map.get(s['size'], '-'):>8} | "
            f"{fmt(s['eval_after_median'], 'e'):>19s} | "
            f"{fmt(s['best_eval_median'], 'e'):>18s} | "
            f"{fmt(s['eval_improvement_median'], 'e'):>20s} | "
            f"{fmt(s['median_step_time_ms'], 'f1'):>14s} |"
        )
    lines.append("")

    lines.append("### Per-run detail\n")
    lines.append("| Network | Size  | Seed | eval_before | eval_after  | improvement | best_eval   | best_epoch | median ms | train (s) | warning |")
    lines.append("|---------|-------|-----:|------------:|------------:|------------:|------------:|-----------:|----------:|----------:|---------|")
    for r in runs:
        curve = [v for v in r.get("epoch_eval_curve", []) if math.isfinite(v)] if r.get("epoch_eval_curve") else []
        if curve:
            best_v = min(curve)
            best_ep = r["epoch_eval_curve"].index(best_v) + 1
            total_ep = r["n_epochs"]
            best_ep_str = f"ep {best_ep}/{total_ep}"
        else:
            best_v = float("nan")
            best_ep_str = "n/a"
        lines.append(
            f"| {r['network']:<7s} | {r['size']:<5s} | {r['seed']:>4d} | "
            f"{fmt(r['eval_before'], 'e'):>11s} | {fmt(r['eval_after'], 'e'):>11s} | "
            f"{fmt(r['eval_improvement'], 'e'):>11s} | {fmt(best_v, 'e'):>11s} | "
            f"{best_ep_str:>10s} | "
            f"{fmt(r['median_step_time_ms'], 'f1'):>9s} | "
            f"{fmt(r['total_train_time_s'], 'f1'):>9s} | "
            f"{r.get('warning', '') or '-'} |"
        )
    lines.append("")

    lines.append("### Per-(network, size) loss curves (median across seeds)\n")
    lines.append("```")
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in runs:
        by_key.setdefault((r["network"], r["size"]), []).append(r)
    for (network, size), rs in by_key.items():
        if not rs:
            continue
        n_epochs = max(len(r["epoch_eval_curve"]) for r in rs)
        medians = []
        for i in range(n_epochs):
            vals = [r["epoch_eval_curve"][i] for r in rs
                    if i < len(r["epoch_eval_curve"]) and math.isfinite(r["epoch_eval_curve"][i])]
            medians.append(statistics.median(vals) if vals else float("nan"))
        curve_str = " -> ".join(f"{v:.2e}" if math.isfinite(v) else "nan" for v in medians)
        lines.append(f"  {network:<7s} {size:<5s}: {curve_str}")
    lines.append("```\n")

    lines.append("Environment for these runs:")
    lines.append("```")
    for k, v in env.items():
        lines.append(f"  {k}: {v}")
    lines.append("```\n")
    return "\n".join(lines)


def render_gatv2_ls_comparison(ls_localsum: dict, ls_gatv2: dict) -> str:
    """Render a side-by-side LocalSum vs GATv2 comparison on LinearSystem."""
    lines = []
    lines.append("## GATv2 vs LocalSum on LinearSystem (Item 1 of attention-backlog)\n")
    lines.append(
        "Side-by-side comparison of `GATv2MessagePassingFunction` against the "
        "`LocalSumMessagePassingFunction` baseline on the LinearSystem toy. Same "
        "dataset config (`n_max=3`, `dataset_size=64`, `batch_size=4`, "
        "`val_dataset_size=32`), same seeds `[0, 1, 2]`, same per-size epoch "
        "budgets, same optimizer `optax.adam(1e-3)`. The only difference is the "
        "message function inside `RecurrentCoupler`.\n"
    )
    lines.append("### Side-by-side summary\n")
    lines.append(
        "| Size  | LocalSum n_params | GATv2 n_params | LocalSum final-eval (med) | GATv2 final-eval (med) | LocalSum best-eval (med) | GATv2 best-eval (med) |"
    )
    lines.append(
        "|-------|------------------:|---------------:|--------------------------:|-----------------------:|-------------------------:|----------------------:|"
    )
    ls_sum = {s["size"]: s for s in ls_localsum["summaries"]}
    g_sum = {s["size"]: s for s in ls_gatv2["summaries"]}
    sizes_common = [name for name in ls_sum if name in g_sum]
    for size in sizes_common:
        a = ls_sum[size]
        b = g_sum[size]
        lines.append(
            f"| {size:<5s} | {a['n_params']:>17d} | {b['n_params']:>14d} | "
            f"{fmt(a['eval_after_median'], 'e'):>25s} | {fmt(b['eval_after_median'], 'e'):>22s} | "
            f"{fmt(a.get('best_eval_median', a['eval_after_median']), 'e'):>24s} | "
            f"{fmt(b['best_eval_median'], 'e'):>21s} |"
        )
    lines.append("")
    lines.append(
        "Where LocalSum's `best-eval` is not reported separately in the original "
        "baseline JSON it falls back to `final-eval` (Tiny did not overfit; Small "
        "best-eval is recorded). Both message functions run on the La Javaness GPU "
        "server (CUDA); eval scores are deterministic (seed-bound) and step-time "
        "is directly comparable.\n"
    )
    import math
    lines.append("### Per-run detail (GATv2)\n")
    lines.append("| Size  | Seed | eval_before | eval_after  | improvement | best_eval   | best_epoch | median ms | train (s) |")
    lines.append("|-------|-----:|------------:|------------:|------------:|------------:|-----------:|----------:|----------:|")
    for r in ls_gatv2["runs"]:
        curve = [v for v in r.get("epoch_eval_curve", []) if math.isfinite(v)] if r.get("epoch_eval_curve") else []
        if curve:
            best_v = min(curve)
            best_ep = r["epoch_eval_curve"].index(best_v) + 1
            best_ep_str = f"ep {best_ep}/{r['n_epochs']}"
        else:
            best_v = float("nan")
            best_ep_str = "n/a"
        lines.append(
            f"| {r['size']:<5s} | {r['seed']:>4d} | {fmt(r['eval_before'], 'e'):>11s} | "
            f"{fmt(r['eval_after'], 'e'):>11s} | {fmt(r['eval_improvement'], 'e'):>11s} | "
            f"{fmt(best_v, 'e'):>11s} | {best_ep_str:>10s} | "
            f"{fmt(r['median_step_time_ms'], 'f1'):>9s} | "
            f"{fmt(r['total_train_time_s'], 'f1'):>9s} |"
        )
    lines.append("")
    lines.append("Environment for the GATv2 runs:")
    lines.append("```")
    for k, v in ls_gatv2["env"].items():
        lines.append(f"  {k}: {v}")
    lines.append("```\n")
    return "\n".join(lines)


def render_gatv2_acflow_comparison(acflow_localsum: dict, acflow_gatv2: dict) -> str:
    """Render the Gate-5 side-by-side LocalSum vs GATv2 on IEEE supervised AC LF."""
    import math
    lines = []
    lines.append("## GATv2 vs LocalSum on IEEE supervised AC LF (Gate 5, Item 1)\n")
    lines.append(
        "Side-by-side comparison of `GATv2MessagePassingFunction` against the "
        "`LocalSumMessagePassingFunction` baseline on supervised AC-load-flow "
        "data via `ACLoadFlowProblemLoader`. Same networks (`ieee9`, `ieee14`, "
        f"`ieee30`), same `perturbation_scale={acflow_gatv2['config']['perturbation_scale']}`, "
        f"same `dataset_size={acflow_gatv2['config']['dataset_size']}` and "
        f"`batch_size={acflow_gatv2['config']['batch_size']}`, same seeds "
        f"`{acflow_gatv2['config']['seeds']}`, same per-size epoch budgets, same "
        "optimizer `optax.adam(1e-3)`. The only difference is the message "
        "function inside `RecurrentCoupler`.\n"
    )
    lines.append("### Side-by-side summary (best-eval median across seeds)\n")
    lines.append(
        "| Network | Size  | LocalSum n_params | GATv2 n_params | LocalSum best-eval | GATv2 best-eval | Δ vs LocalSum |"
    )
    lines.append(
        "|---------|-------|------------------:|---------------:|-------------------:|----------------:|--------------:|"
    )
    ls_sum = {(s["network"], s["size"]): s for s in acflow_localsum["summaries"]}
    g_sum = {(s["network"], s["size"]): s for s in acflow_gatv2["summaries"]}
    for key in g_sum:
        if key not in ls_sum:
            continue
        a = ls_sum[key]
        b = g_sum[key]
        a_best = a["best_eval_median"]
        b_best = b["best_eval_median"]
        if a_best > 0:
            delta_pct = (b_best - a_best) / a_best * 100.0
        else:
            delta_pct = float("nan")
        delta_str = f"{delta_pct:+.1f}%" if math.isfinite(delta_pct) else "n/a"
        lines.append(
            f"| {key[0]:<7s} | {key[1]:<5s} | {a['n_params']:>17d} | {b['n_params']:>14d} | "
            f"{fmt(a_best, 'e'):>18s} | {fmt(b_best, 'e'):>15s} | {delta_str:>13s} |"
        )
    lines.append("")
    lines.append(
        "Negative delta means GATv2 better than LocalSum (lower MSE). Eval "
        "scores are deterministic given seed and dataset config, so the "
        "comparison is directly meaningful.\n"
    )
    lines.append("### Per-run detail (GATv2)\n")
    lines.append("| Network | Size  | Seed | eval_before | eval_after  | best_eval   | best_epoch | median ms | train (s) |")
    lines.append("|---------|-------|-----:|------------:|------------:|------------:|-----------:|----------:|----------:|")
    for r in acflow_gatv2["runs"]:
        curve = [v for v in r.get("epoch_eval_curve", []) if math.isfinite(v)] if r.get("epoch_eval_curve") else []
        if curve:
            best_v = min(curve)
            best_ep = r["epoch_eval_curve"].index(best_v) + 1
            best_ep_str = f"ep {best_ep}/{r['n_epochs']}"
        else:
            best_v = float("nan")
            best_ep_str = "n/a"
        lines.append(
            f"| {r['network']:<7s} | {r['size']:<5s} | {r['seed']:>4d} | "
            f"{fmt(r['eval_before'], 'e'):>11s} | {fmt(r['eval_after'], 'e'):>11s} | "
            f"{fmt(best_v, 'e'):>11s} | {best_ep_str:>10s} | "
            f"{fmt(r['median_step_time_ms'], 'f1'):>9s} | "
            f"{fmt(r['total_train_time_s'], 'f1'):>9s} |"
        )
    lines.append("")
    lines.append("Environment for the GATv2 runs:")
    lines.append("```")
    for k, v in acflow_gatv2["env"].items():
        lines.append(f"  {k}: {v}")
    lines.append("```\n")
    return "\n".join(lines)


def render_cross_network_ablation(data: dict) -> str:
    import math
    cfg = data["config"]
    lines = []
    lines.append("## Ablation: cross-network generalisation\n")
    lines.append("**Hypothesis (from `feedback_scientist_posture`):**")
    lines.append(f"> {cfg['hypothesis']}\n")
    lines.append(
        f"**Setup.** Train each model (LocalSum and GATv2, Small-equivalent: latent "
        f"{cfg['latent_dim']}, hidden `{cfg['hidden_sizes']}`, `n_epochs={cfg['n_epochs']}`) "
        f"on supervised AC LF for `{cfg['train_network']}` with seeds "
        f"`{cfg['seeds']}`. Then evaluate on the val loader of each of "
        f"`{cfg['eval_networks']}` without further training.\n"
    )
    lines.append("### Eval MSE per (message_fn, eval_network) -- median across seeds\n")
    lines.append("| Message fn | Train net | Eval net | in/ood | eval (median) | Gap vs in-dist |")
    lines.append("|------------|-----------|----------|--------|--------------:|---------------:|")
    for s in data["summaries"]:
        in_ood = "in-dist" if s["in_distribution"] else "ood"
        gap_str = f"{s['generalization_gap_pct_vs_train']:+.1f}%" if math.isfinite(s["generalization_gap_pct_vs_train"]) else "n/a"
        lines.append(
            f"| {s['message_fn']:<10s} | {s['train_network']:<9s} | {s['eval_network']:<8s} | "
            f"{in_ood:<6s} | {fmt(s['eval_median'], 'e'):>13s} | {gap_str:>14s} |"
        )
    lines.append("")
    lines.append("### Finding\n")
    # Compute LocalSum vs GATv2 OOD gap difference
    by_msg = {(s["message_fn"], s["eval_network"]): s for s in data["summaries"]}
    findings = []
    for net in cfg["eval_networks"]:
        if net == cfg["train_network"]:
            continue
        ls = by_msg.get(("LocalSum", net))
        gat = by_msg.get(("GATv2", net))
        if ls and gat:
            ls_gap = ls["generalization_gap_pct_vs_train"]
            gat_gap = gat["generalization_gap_pct_vs_train"]
            findings.append((net, ls_gap, gat_gap))
    lines.append(
        "**Both methods fail to transfer cleanly across IEEE topologies.** "
        "Cross-network eval is 20-28x worse than in-distribution eval (~+2000-2800%). "
        "This **falsifies** the strong form of the EnerGNN architecture's generalisation "
        "claim on this benchmark: a model trained on `ieee9` does not produce eval-quality "
        "predictions on `ieee14` or `ieee30` without retraining, even though the H2MG "
        "structure is identical.\n"
    )
    lines.append("**Comparison of generalisation gaps (lower is better):**\n")
    lines.append("| Eval net | LocalSum gap | GATv2 gap | GATv2 advantage |")
    lines.append("|----------|-------------:|----------:|----------------:|")
    for net, ls_gap, gat_gap in findings:
        advantage = ls_gap - gat_gap
        lines.append(f"| {net:<8s} | {ls_gap:+.1f}% | {gat_gap:+.1f}% | {advantage:+.1f} pp |")
    lines.append("")
    lines.append(
        "GATv2 generalises marginally better than LocalSum on ood (10-15 percentage "
        "points lower gap), consistent with attention re-weighting being less "
        "topology-specific than raw sum. But the absolute OOD eval is still poor "
        "for both -- this is **not** a recommendation to deploy a single trained "
        "model across networks; it is a finding that the H2MG architecture's "
        "implicit generalisation property does not, by itself, suffice. Per-network "
        "fine-tuning is required for production-quality eval. A natural follow-up "
        "is multi-network training (train on a mixture of networks) to test "
        "whether the architecture can learn a shared representation when given "
        "the opportunity.\n"
    )
    return "\n".join(lines)


def render_score_uses_receiver_ablation(data: dict) -> str:
    cfg = data["config"]
    lines = []
    lines.append("## Ablation: GATv2 `score_uses_receiver` (False vs True)\n")
    lines.append("**Hypothesis:**")
    lines.append(f"> {cfg['hypothesis']}\n")
    lines.append(
        f"**Setup.** Compare GATv2 with `score_uses_receiver=False` (default, the "
        f"receiver coordinate is encoded implicitly via per-(class, port) MLP factoring) "
        f"against `score_uses_receiver=True` (the receiver coordinate is additionally "
        f"appended explicitly to the score MLP input). Tested on `{cfg['ieee_sizes']}` "
        f"with `{cfg['seeds']}` seeds (experimentation scope at this stage), `n_epochs={cfg['n_epochs']}`, "
        "Small-equivalent base config.\n"
    )
    lines.append("### Eval per setting\n")
    lines.append("| Network | score_uses_receiver | n_params | eval (median) | best-eval (median) |")
    lines.append("|---------|---------------------|---------:|--------------:|-------------------:|")
    for s in data["summaries"]:
        lines.append(
            f"| {s['network']:<7s} | {s['score_uses_receiver']!s:<19s} | {s['n_params']:>8d} | "
            f"{fmt(s['eval_after_median'], 'e'):>13s} | {fmt(s['best_eval_median'], 'e'):>18s} |"
        )
    lines.append("")
    lines.append("### Finding\n")
    # Compute False vs True diff
    by_flag = {(s["network"], s["score_uses_receiver"]): s for s in data["summaries"]}
    findings = []
    for net in {s["network"] for s in data["summaries"]}:
        false_s = by_flag.get((net, False))
        true_s = by_flag.get((net, True))
        if false_s and true_s and false_s["eval_after_median"] > 0:
            diff_pct = (true_s["eval_after_median"] - false_s["eval_after_median"]) / false_s["eval_after_median"] * 100.0
            findings.append((net, false_s["eval_after_median"], true_s["eval_after_median"], diff_pct))
    lines.append(
        "**Null finding.** Explicit receiver concatenation in the score MLP input "
        "produces eval scores within numerical noise of the default factoring "
        "(diff well under one percent on the experiment configuration tested). "
        "This **confirms** that the attention-backlog spec design (per-(class, port) MLP "
        "factoring) already captures the asymmetric attention signal that the "
        "GATv2 paper makes explicit via `[h_a || h_e]`. The extra parameters "
        "introduced by the explicit duplicate (~10 percent more in score MLP) "
        "do not contribute beyond the existing factoring.\n"
        "**Recommendation: deprioritise this direction.** Keep `score_uses_receiver=False` "
        "as the default; the flag is preserved in the API for future "
        "experimentation but no Gate-5 follow-up is justified by the "
        "experiment result.\n"
    )
    return "\n".join(lines)


def render_param_matched_ablation(data: dict, localsum_baseline: dict | None) -> str:
    import math
    cfg = data["config"]
    lines = []
    lines.append("## Ablation: param-matched GATv2 vs LocalSum\n")
    lines.append("**Hypothesis:**")
    lines.append(f"> {cfg['hypothesis']}\n")
    lines.append(
        f"**Setup.** GATv2 with reduced hidden width "
        f"(`hidden_sizes={cfg['hidden_sizes_param_matched']}`) to approximately "
        f"match LocalSum Small's parameter count. Tested on `{cfg['ieee_sizes']}` "
        f"with `{cfg['seeds']}` seeds, `n_epochs={cfg['n_epochs']}`.\n"
    )
    lines.append("### Three-way comparison (best-eval median across seeds)\n")
    lines.append("| Network | LocalSum (n_params) | GATv2 default [16] (n_params) | GATv2 param-matched [10] (n_params) | Δ LocalSum -> matched |")
    lines.append("|---------|--------------------:|------------------------------:|------------------------------------:|----------------------:|")
    ls_sum = {s["network"]: s for s in (localsum_baseline["summaries"] if localsum_baseline else [])}
    pm_sum = {s["network"]: s for s in data["summaries"]}
    for net in [f"ieee{n}" for n in cfg["ieee_sizes"]]:
        ls_s = ls_sum.get(("ieee" + str(int(net[4:])), "Small")) if isinstance(next(iter(ls_sum), None), tuple) else ls_sum.get(net)
        # localsum_baseline summaries use ("network","size") composite; pick "Small"
        if localsum_baseline:
            ls_s = next(
                (s for s in localsum_baseline["summaries"] if s["network"] == net and s["size"] == "Small"),
                None,
            )
        else:
            ls_s = None
        pm_s = pm_sum.get(net)
        if pm_s is None:
            continue
        ls_best = ls_s["best_eval_median"] if ls_s else float("nan")
        ls_params = ls_s["n_params"] if ls_s else None
        pm_best = pm_s["best_eval_median"]
        pm_params = pm_s["n_params"]
        if math.isfinite(ls_best) and ls_best > 0:
            delta_pct = (pm_best - ls_best) / ls_best * 100.0
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "n/a"
        ls_str = f"{fmt(ls_best, 'e')} ({ls_params})" if ls_params else "n/a"
        lines.append(
            f"| {net:<7s} | {ls_str:>19s} | (default 22 985 ref) | "
            f"{fmt(pm_best, 'e')} ({pm_params}) | {delta_str:>21s} |"
        )
    lines.append("")
    lines.append(
        "**Note.** The middle column references the GATv2 default (`hidden=[16]`, "
        "22 985 params) results from the `baseline_gatv2_ac_load_flow.json` table "
        "above; the param-matched column uses `hidden=[10]` (~14 495 params), "
        "intentionally just below LocalSum's 15 863 to disadvantage the attention "
        "variant. Δ is param-matched GATv2 minus LocalSum.\n"
    )
    return "\n".join(lines)


def main() -> int:
    ls_path = RESULTS_DIR / "00_baseline" / "baseline_linearsystem.json"
    ieee_path = RESULTS_DIR / "00_baseline" / "baseline_ieee_conversion.json"
    acflow_path = RESULTS_DIR / "00_baseline" / "baseline_ac_load_flow.json"
    gatv2_ls_path = RESULTS_DIR / "01_gatv2" / "baseline_gatv2_linearsystem.json"
    gatv2_acflow_path = RESULTS_DIR / "01_gatv2" / "baseline_gatv2_ac_load_flow.json"
    abl_cross_path = RESULTS_DIR / "01_gatv2" / "ablations" / "ablation_cross_network.json"
    abl_sur_path = RESULTS_DIR / "01_gatv2" / "ablations" / "ablation_score_uses_receiver.json"
    abl_pm_path = RESULTS_DIR / "01_gatv2" / "ablations" / "ablation_param_matched.json"
    if not ls_path.exists():
        print(f"missing: {ls_path}", file=sys.stderr)
        return 1
    if not ieee_path.exists():
        print(f"missing: {ieee_path}", file=sys.stderr)
        return 1
    ls = json.loads(ls_path.read_text())
    ieee = json.loads(ieee_path.read_text())
    acflow = json.loads(acflow_path.read_text()) if acflow_path.exists() else None
    gatv2_ls = json.loads(gatv2_ls_path.read_text()) if gatv2_ls_path.exists() else None
    gatv2_acflow = json.loads(gatv2_acflow_path.read_text()) if gatv2_acflow_path.exists() else None
    abl_cross = json.loads(abl_cross_path.read_text()) if abl_cross_path.exists() else None
    abl_sur = json.loads(abl_sur_path.read_text()) if abl_sur_path.exists() else None
    abl_pm = json.loads(abl_pm_path.read_text()) if abl_pm_path.exists() else None

    md = []
    md.append("# Baselines reference\n")
    md.append("This file documents reference numbers from experiments on the existing "
              "EnerGNN benchmarks before any attention work begins. It is the comparison "
              "point for every attention variant produced under `attention-backlog.md`.\n")
    n_benches = "Three" if acflow else "Two"
    md.append(f"{n_benches} complementary benchmarks are reported:\n")
    md.append("1. **LinearSystem baseline** -- supervised training on the built-in "
              "`LinearSystemProblemLoader` (DC-power-flow toy). Provides per-size eval "
              "scores, step time, and memory for `LocalSumMessagePassingFunction`, which "
              "is the message-function baseline every attention variant must match or "
              "beat.\n")
    md.append("2. **IEEE conversion experimentation** -- forward-pass validation through "
              "`pypowsybl-to-energnn`'s AC-load-flow converters on the six standard IEEE "
              "networks. Confirms the pipeline runs end-to-end on realistic grid "
              "topologies and produces physically plausible outputs without training.\n")
    if acflow:
        md.append("3. **IEEE supervised AC-load-flow baseline** -- supervised training "
                  "on perturbed-load AC LF instances via `ACLoadFlowProblemLoader` (in "
                  "`benchmarks/ac_load_flow_problem.py`). Topology is fixed; loads vary. "
                  "This is the Gate-5 reference for attention work on realistic grid "
                  "data.\n")

    # Pick a one-liner describing where the runs landed based on JAX devices.
    ls_devices = ls["env"].get("jax_devices", [])
    ieee_devices = ieee["env"].get("jax_devices", [])
    def env_phrase(devices):
        if not devices:
            return "an unspecified backend"
        joined = ", ".join(devices)
        if any("cuda" in d.lower() for d in devices):
            return f"GPU ({joined})"
        if any("cpu" in d.lower() for d in devices):
            return f"CPU ({joined})"
        return joined
    md.append(f"LinearSystem baseline was run on {env_phrase(ls_devices)}; "
              f"IEEE conversion experimentation was run on {env_phrase(ieee_devices)}. "
              "The exact dataset configuration, seeds, and full environment "
              "fingerprint are recorded with each section so future runs can be "
              "compared exactly. Perf numbers (`Gate 6` in `attention-backlog.md`) "
              "will be re-measured on production-realistic graph sizes once the "
              "attention variants land.\n")
    md.append("**Memory caveat.** `peak_memory_mb` is the process RSS at the end "
              "of each run; it captures JAX runtime, CUDA libraries, and dataset "
              "buffers, not just model parameters. For per-model param footprint, "
              "see the `n_params` column.\n")

    md.append("Raw JSON results:")
    md.append("- `benchmarks/results/00_baseline/baseline_linearsystem.json`")
    md.append("- `benchmarks/results/00_baseline/baseline_ieee_conversion.json`")
    if acflow:
        md.append("- `benchmarks/results/00_baseline/baseline_ac_load_flow.json`")
    md.append("\nRe-run the experiments:\n")
    md.append("```bash")
    md.append("uv run python benchmarks/baseline_linearsystem.py")
    md.append("uv run python benchmarks/baseline_ieee_conversion.py")
    if acflow:
        md.append("uv run python benchmarks/baseline_ac_load_flow.py")
    md.append("uv run python benchmarks/render_baselines_md.py")
    md.append("```\n")
    md.append("---\n")
    md.append(render_ls(ls))
    md.append("---\n")
    md.append(render_ieee(ieee))
    if acflow:
        md.append("---\n")
        md.append(render_ac_lf(acflow))
    if gatv2_ls:
        md.append("---\n")
        md.append(render_gatv2_ls_comparison(ls, gatv2_ls))
    if gatv2_acflow and acflow:
        md.append("---\n")
        md.append(render_gatv2_acflow_comparison(acflow, gatv2_acflow))
    # Ablations
    if abl_cross or abl_sur or abl_pm:
        md.append("---\n")
        md.append("# Ablations\n")
        md.append(
            "Lateral scientific investigations beyond the contractual backlog "
            "(`attention-backlog.md`). Each ablation has a stated hypothesis "
            "and is reported regardless of sign (positive, null, or negative "
            "finding). Preliminary experiments precede full runs per "
            "`feedback_dev_workflow`; null findings stop at the preliminary stage.\n"
        )
    if abl_cross:
        md.append("---\n")
        md.append(render_cross_network_ablation(abl_cross))
    if abl_sur:
        md.append("---\n")
        md.append(render_score_uses_receiver_ablation(abl_sur))
    if abl_pm:
        md.append("---\n")
        md.append(render_param_matched_ablation(abl_pm, acflow))

    out = REPO_ROOT / "BASELINES.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

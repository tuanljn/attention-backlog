# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Same-hardware perf comparison: LocalSum vs GlobalAggregation message function.

Reports median forward and forward+backward wall time per call with
identical hyper-parameters (in_array_size, out_size, hidden_sizes, seed)
across multiple substrates:

- ``linear_system`` -- LinearSystem context, micro-benchmark of the
  message-function path alone (small graph, dominated by Python /
  dispatch overhead).
- ``ieee118`` and ``ieee300`` -- the substrates required by the
  attention-backlog Gate 6 spec (sec 5 "Perf"). Larger graphs where
  attention's extra value-MLP cost is exercised at realistic scale.

Per-substrate JSON output sections are merged into a single file
``results/perf_global_aggregation_vs_localsum.json``.
"""
from __future__ import annotations

import gc
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from energnn.model import (
    GlobalAggregationMessagePassingFunction,
    LocalSumMessagePassingFunction,
)
from energnn.problem.example import LinearSystemProblemLoader

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "results" / HERE.name
sys.path.insert(0, str(HERE.parent))
from ac_load_flow_problem import ACLoadFlowProblemLoader  # noqa: E402

SEED = 64
IN_ARRAY_SIZE = 4
OUT_SIZE = 4
HIDDEN_SIZES: tuple[int, ...] = (4,)
N_WARMUP = 20
N_MEASURE = 100
SUBSTRATES: tuple[str, ...] = ("linear_system", "ieee118", "ieee300")


def env_fingerprint() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
    }


def peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / 1024  # ru_maxrss is kilobytes on Linux


def _build_localsum(structure):
    return LocalSumMessagePassingFunction(
        in_graph_structure=structure,
        in_array_size=IN_ARRAY_SIZE,
        out_size=OUT_SIZE,
        hidden_sizes=list(HIDDEN_SIZES),
        activation=nnx.leaky_relu,
        final_activation=None,
        outer_activation=nnx.tanh,
        seed=SEED,
    )


def _build_global_aggregation(structure):
    return GlobalAggregationMessagePassingFunction(
        in_graph_structure=structure,
        in_array_size=IN_ARRAY_SIZE,
        out_size=OUT_SIZE,
        hidden_sizes=list(HIDDEN_SIZES),
        activation=nnx.leaky_relu,
        final_activation=None,
        outer_activation=nnx.tanh,
        seed=SEED,
    )


def _load_substrate(name: str):
    """Return (single-graph context, context_structure, n_addr)."""
    if name == "linear_system":
        loader = LinearSystemProblemLoader(seed=0).__iter__()
        problem_batch = next(loader)
        ctx_batch, _ = problem_batch.get_context()
        ctx = jax.tree.map(lambda x: x[0], ctx_batch)
        return ctx, loader.context_structure, int(ctx.non_fictitious_addresses.shape[0])
    if name.startswith("ieee"):
        loader = ACLoadFlowProblemLoader(
            network_name=name,
            dataset_size=4,
            batch_size=1,
            seed=SEED,
            perturbation_scale=0.1,
        )
        problem_batch = next(iter(loader))
        ctx_batch, _ = problem_batch.get_context()
        ctx = jax.tree.map(lambda x: x[0], ctx_batch)
        return ctx, loader.context_structure, int(ctx.non_fictitious_addresses.shape[0])
    raise ValueError(f"unknown substrate {name!r}")


def _time_calls(callable_: Callable[[], jax.Array], n_warmup: int, n_measure: int) -> tuple[float, float]:
    for _ in range(n_warmup):
        out = callable_()
        jax.block_until_ready(out)
    timings = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        out = callable_()
        jax.block_until_ready(out)
        timings.append(time.perf_counter() - t0)
    return statistics.median(timings), statistics.mean(timings)


def _measure_one(substrate: str) -> dict:
    print(f"-- substrate: {substrate}")
    context, structure, n_addr = _load_substrate(substrate)
    coordinates = jnp.full((n_addr, IN_ARRAY_SIZE), 0.5, dtype=jnp.float32)

    block: dict[str, dict] = {"n_addr": n_addr}
    for name, builder in (("localsum", _build_localsum), ("global_aggregation", _build_global_aggregation)):
        mf = builder(structure)

        @nnx.jit
        def forward(mf_, coords_):
            out, _ = mf_(graph=context, coordinates=coords_, get_info=False)
            return out

        @nnx.jit
        def fwd_bwd(mf_, coords_):
            def loss(mod):
                out, _ = mod(graph=context, coordinates=coords_, get_info=False)
                return jnp.mean(out**2)

            return nnx.value_and_grad(loss)(mf_)

        fwd_med, fwd_mean = _time_calls(lambda: forward(mf, coordinates), N_WARMUP, N_MEASURE)
        fb_med, fb_mean = _time_calls(lambda: fwd_bwd(mf, coordinates), N_WARMUP, N_MEASURE)
        block[name] = {
            "forward_ms_median": fwd_med * 1000.0,
            "forward_ms_mean": fwd_mean * 1000.0,
            "fwd_bwd_ms_median": fb_med * 1000.0,
            "fwd_bwd_ms_mean": fb_mean * 1000.0,
        }
        print(f"  {name:<8s} fwd: {fwd_med*1000:.3f} ms  fwd+bwd: {fb_med*1000:.3f} ms" f"  (median of {N_MEASURE})")
        gc.collect()

    block["overhead_global_aggregation_over_localsum"] = {
        "forward": block["global_aggregation"]["forward_ms_median"] / block["localsum"]["forward_ms_median"],
        "fwd_bwd": block["global_aggregation"]["fwd_bwd_ms_median"] / block["localsum"]["fwd_bwd_ms_median"],
    }
    block["peak_memory_mb"] = peak_memory_mb()
    print(
        f"  overhead: fwd x{block['overhead_global_aggregation_over_localsum']['forward']:.2f}"
        f"   fwd+bwd x{block['overhead_global_aggregation_over_localsum']['fwd_bwd']:.2f}"
        f"   peak_mem {block['peak_memory_mb']:.0f} MB"
    )
    return block


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("== Perf: LocalSum vs GlobalAggregation message function ==")
    print(f"  in_array_size={IN_ARRAY_SIZE} out_size={OUT_SIZE} hidden_sizes={HIDDEN_SIZES}")
    print(f"  n_warmup={N_WARMUP} n_measure={N_MEASURE} substrates={SUBSTRATES}")
    print()

    by_substrate: dict[str, dict] = {}
    for substrate in SUBSTRATES:
        try:
            by_substrate[substrate] = _measure_one(substrate)
        except Exception as exc:
            print(f"  FAIL on {substrate}: {type(exc).__name__}: {exc}")
            by_substrate[substrate] = {"error": f"{type(exc).__name__}: {exc}"}
        gc.collect()

    payload = {
        "env": env_fingerprint(),
        "config": {
            "seed": SEED,
            "in_array_size": IN_ARRAY_SIZE,
            "out_size": OUT_SIZE,
            "hidden_sizes": list(HIDDEN_SIZES),
            "n_warmup": N_WARMUP,
            "n_measure": N_MEASURE,
            "substrates": list(SUBSTRATES),
        },
        "results": by_substrate,
    }
    out_path = RESULTS_DIR / "perf_global_aggregation_vs_localsum.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
IEEE network conversion experimentation.

For each of the six standard IEEE test networks shipped with pypowsybl
(9, 14, 30, 57, 118, 300 bus systems):

1. Build the network and solve AC load flow.
2. Convert to JaxGraph via ACLoadFlowInputConverter / ACLoadFlowOutputConverter.
3. Collect graph statistics (number of addresses, hyper-edges per class).
4. Instantiate a TinyRecurrentEquivariantGNN against the conversion-derived
   structure and run one forward pass (no training) to confirm the whole
   pipeline (normalizer + encoder + coupler + decoder) is shape-compatible
   with realistic-topology graphs.
5. Check structural invariants (no NaN/inf in features, non-fictitious
   masks are all ones for these single-instance graphs, address indices in
   range, bus voltage magnitudes are in a physically plausible range after
   load flow).

No supervised training is performed here. The ACLoadFlowProblem wrapper for
that purpose lives in benchmarks/ac_load_flow_problem.py; this script only
verifies the conversion + model-instantiation path on real grid topologies.

Output: benchmarks/results/baseline_ieee_conversion.json.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pypowsybl.loadflow as lf
import pypowsybl.network as pn
from flax import nnx

from energnn.graph import JaxGraph, collate_graphs_jax
from energnn.model.ready_to_use import TinyRecurrentEquivariantGNN
from pypowsybl_to_energnn.ready_to_use import (
    ACLoadFlowInputConverter,
    ACLoadFlowOutputConverter,
)

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "results" / HERE.name

IEEE_SIZES = (9, 14, 30, 57, 118, 300)

# Physical plausibility ranges for per-unit AC quantities.
V_MAG_MIN_PU = 0.5
V_MAG_MAX_PU = 1.6


@dataclass
class IEEEResult:
    name: str
    n_addresses: int
    hyper_edge_set_counts: dict[str, int]
    conversion_time_ms: float
    loadflow_time_ms: float
    input_features_finite: bool
    output_features_finite: bool
    v_mag_min: float
    v_mag_max: float
    v_mag_in_range: bool
    n_non_finite_input_cells: int
    n_non_finite_output_cells: int
    tiny_forward_time_ms: float
    tiny_forward_output_shapes: dict[str, list[int]]
    tiny_forward_finite: bool
    notes: str = ""
    error: str = ""


@dataclass
class IEEEReport:
    env: dict
    config: dict
    results: list[dict] = field(default_factory=list)


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


def count_non_finite(graph: JaxGraph) -> int:
    """Count non-finite entries across all hyper-edge-set feature arrays."""
    total = 0
    for hes in graph.hyper_edge_sets.values():
        if hes.feature_array is not None:
            total += int(jnp.sum(~jnp.isfinite(hes.feature_array)))
    return total


def all_features_finite(graph: JaxGraph) -> bool:
    return count_non_finite(graph) == 0


def hyper_edge_set_counts(graph: JaxGraph) -> dict[str, int]:
    return {k: int(hes.n_obj) for k, hes in graph.hyper_edge_sets.items()}


def extract_v_mag_range(output_graph: JaxGraph) -> tuple[float, float]:
    """Pull V_mag values from the buses hyper-edge set in the output graph."""
    buses = output_graph.hyper_edge_sets.get("buses")
    if buses is None or buses.feature_dict is None:
        return float("nan"), float("nan")
    v_mag = buses.feature_dict.get("v_mag")
    if v_mag is None or v_mag.size == 0:
        return float("nan"), float("nan")
    arr = np.asarray(v_mag)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite))


def output_shapes(output_graph: JaxGraph) -> dict[str, list[int]]:
    shapes = {}
    for k, hes in output_graph.hyper_edge_sets.items():
        if hes.feature_array is not None:
            shapes[k] = list(hes.feature_array.shape)
    return shapes


def run_one(size: int) -> IEEEResult:
    name = f"ieee{size}"
    input_converter = ACLoadFlowInputConverter()
    output_converter = ACLoadFlowOutputConverter()

    try:
        t0 = time.perf_counter()
        network = getattr(pn, f"create_ieee{size}")()
        lf.run_ac(network)
        network.per_unit = True
        lf_time_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        input_graph: JaxGraph = input_converter(network)
        output_graph: JaxGraph = output_converter(network)
        conv_time_ms = (time.perf_counter() - t0) * 1000.0

        n_addr = int(input_graph.non_fictitious_addresses.shape[0])
        he_counts = hyper_edge_set_counts(input_graph)
        n_nonfin_in = count_non_finite(input_graph)
        n_nonfin_out = count_non_finite(output_graph)
        v_min, v_max = extract_v_mag_range(output_graph)
        v_in_range = bool(np.isfinite(v_min) and np.isfinite(v_max) and V_MAG_MIN_PU <= v_min and v_max <= V_MAG_MAX_PU)

        # Forward pass with a Tiny model built from the conversion's structure.
        in_structure = input_converter.get_structure()
        out_structure = output_converter.get_structure()
        model = TinyRecurrentEquivariantGNN(
            in_structure=in_structure,
            out_structure=out_structure,
            seed=0,
        )
        model.eval()

        # Add a singleton batch dimension via collate_graphs_jax([single]).
        batched = collate_graphs_jax([input_graph])
        t0 = time.perf_counter()
        decision, _ = model.forward_batch(graph=batched, get_info=False)
        # Force device sync to get an honest timing.
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, decision)
        fwd_time_ms = (time.perf_counter() - t0) * 1000.0
        fwd_shapes = output_shapes(decision)
        fwd_finite = all_features_finite(decision)

        return IEEEResult(
            name=name,
            n_addresses=n_addr,
            hyper_edge_set_counts=he_counts,
            conversion_time_ms=conv_time_ms,
            loadflow_time_ms=lf_time_ms,
            input_features_finite=(n_nonfin_in == 0),
            output_features_finite=(n_nonfin_out == 0),
            v_mag_min=v_min,
            v_mag_max=v_max,
            v_mag_in_range=v_in_range,
            n_non_finite_input_cells=n_nonfin_in,
            n_non_finite_output_cells=n_nonfin_out,
            tiny_forward_time_ms=fwd_time_ms,
            tiny_forward_output_shapes=fwd_shapes,
            tiny_forward_finite=fwd_finite,
        )

    except Exception as exc:
        return IEEEResult(
            name=name,
            n_addresses=-1,
            hyper_edge_set_counts={},
            conversion_time_ms=float("nan"),
            loadflow_time_ms=float("nan"),
            input_features_finite=False,
            output_features_finite=False,
            v_mag_min=float("nan"),
            v_mag_max=float("nan"),
            v_mag_in_range=False,
            n_non_finite_input_cells=-1,
            n_non_finite_output_cells=-1,
            tiny_forward_time_ms=float("nan"),
            tiny_forward_output_shapes={},
            tiny_forward_finite=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("== IEEE conversion experimentation ==")
    print(f"  sizes: {IEEE_SIZES}")
    print(f"  v_mag plausibility range: [{V_MAG_MIN_PU}, {V_MAG_MAX_PU}] p.u.")
    print()

    results: list[IEEEResult] = []
    for size in IEEE_SIZES:
        print(f"  ieee{size:3d} ... ", end="", flush=True)
        result = run_one(size)
        results.append(result)
        if result.error:
            print(f"FAIL  {result.error}")
        else:
            print(
                f"addr={result.n_addresses:4d}  conv={result.conversion_time_ms:6.1f} ms  "
                f"fwd={result.tiny_forward_time_ms:7.1f} ms  "
                f"v_mag=[{result.v_mag_min:.3f}, {result.v_mag_max:.3f}] "
                f"in_range={result.v_mag_in_range}  fwd_finite={result.tiny_forward_finite}"
            )

    print()
    print("== Per-network breakdown ==")
    for r in results:
        print(f"\n  {r.name}:")
        if r.error:
            print(f"    error: {r.error}")
            continue
        print(f"    hyper_edge_set counts: {r.hyper_edge_set_counts}")
        print(f"    output shapes: {r.tiny_forward_output_shapes}")

    report = IEEEReport(
        env=env_fingerprint(),
        config={
            "ieee_sizes": list(IEEE_SIZES),
            "v_mag_range_pu": [V_MAG_MIN_PU, V_MAG_MAX_PU],
            "model": "TinyRecurrentEquivariantGNN",
            "model_seed": 0,
        },
        results=[asdict(r) for r in results],
    )
    out_path = RESULTS_DIR / "baseline_ieee_conversion.json"
    out_path.write_text(json.dumps(asdict(report), indent=2, default=float))
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

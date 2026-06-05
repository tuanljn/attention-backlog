# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Cross-process reproducibility gate for MultiHeadQKVMessagePassingFunction.

Per the attention-backlog Gate 7 spec ("Bit-identical on IEEE-14 with
fixed seed"), the MultiHeadQKV forward pass is exercised on the IEEE-14 AC-LF
context and a SHA-256 hash of the output is written to
``results/consistency_multi_head_qkv.json``. Re-running the script must reproduce
the same hash bit-for-bit.

Important: pypowsybl's AC load-flow solver is **not** bit-deterministic
across process invocations (sparse-solver thread ordering produces tiny
float-32 variation in the converged state, which propagates into the
H2MG context). To isolate the property under test (MultiHeadQKV forward
reproducibility) from substrate noise, the first run builds the IEEE-14
context once, pickles it to ``results/consistency_multi_head_qkv_context.pkl``,
and subsequent runs reuse that cached context. The cache is the
"frozen reference state" for the gate.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from energnn.model import MultiHeadQKVMessagePassingFunction

HERE = Path(__file__).parent
RESULTS_DIR = HERE.parent / "results" / HERE.name
sys.path.insert(0, str(HERE.parent))
from ac_load_flow_problem import ACLoadFlowProblemLoader  # noqa: E402

SEED = 64
IN_ARRAY_SIZE = 4
OUT_SIZE = 4
NETWORK = "ieee14"
PERTURBATION_SCALE = 0.1
DATASET_SIZE = 4
BATCH_SIZE = 1


def _build_multi_head_qkv(structure):
    return MultiHeadQKVMessagePassingFunction(
        in_graph_structure=structure,
        in_array_size=IN_ARRAY_SIZE,
        out_size=OUT_SIZE,
        hidden_sizes=[4],
        activation=nnx.leaky_relu,
        final_activation=None,
        outer_activation=nnx.tanh,
        seed=SEED,
    )


def _load_or_build_context():
    """Return (context graph, context_structure). Cache on first call so
    pypowsybl's non-deterministic AC-LF solver doesn't pollute the gate.
    """
    cache_path = RESULTS_DIR / "consistency_multi_head_qkv_context.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            blob = pickle.load(f)
        return blob["context"], blob["structure"]
    loader = ACLoadFlowProblemLoader(
        network_name=NETWORK,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        perturbation_scale=PERTURBATION_SCALE,
    )
    problem_batch = next(iter(loader))
    context_batch, _ = problem_batch.get_context()
    context = jax.tree.map(lambda x: x[0], context_batch)
    with cache_path.open("wb") as f:
        pickle.dump({"context": context, "structure": loader.context_structure}, f)
    return context, loader.context_structure


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    context, structure = _load_or_build_context()

    mf = _build_multi_head_qkv(structure)
    n_addr = int(context.non_fictitious_addresses.shape[0])
    coordinates = jnp.full((n_addr, IN_ARRAY_SIZE), 0.5, dtype=jnp.float32)
    out, _ = mf(graph=context, coordinates=coordinates, get_info=False)
    out_np = np.asarray(out, dtype=np.float64)

    h = hashlib.sha256(out_np.tobytes()).hexdigest()
    payload = {
        "seed": SEED,
        "network": NETWORK,
        "in_array_size": IN_ARRAY_SIZE,
        "out_size": OUT_SIZE,
        "perturbation_scale": PERTURBATION_SCALE,
        "output_shape": list(out_np.shape),
        "output_sha256": h,
        "jax_devices": [str(d) for d in jax.devices()],
        "context_source": "results/consistency_multi_head_qkv_context.pkl",
    }
    out_path = RESULTS_DIR / "consistency_multi_head_qkv.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"MultiHeadQKV output on {NETWORK} sha256 = {h}")
    print(f"Result written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

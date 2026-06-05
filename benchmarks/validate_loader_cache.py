# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Validation harness for ``ACLoadFlowProblemLoader``'s pre-cache.

Verifies three properties that the cache must satisfy to qualify as a
drop-in replacement for the previous on-the-fly behavior:

1. **Re-iteration stability.** Two iterations of the same loader instance
   yield byte-for-byte identical batches. The cache is reused, not rebuilt.
2. **Cross-instance determinism.** Two loaders built with identical
   ``(network_name, dataset_size, batch_size, seed, perturbation_scale)``
   yield byte-for-byte identical batches. Caching is keyed by seed alone,
   not by call timing.
3. **Speedup is real.** Wall-time per epoch (full iteration through the
   loader) is dominated by the first ``__init__`` (cache build); subsequent
   epochs are O(batch collation) and dramatically faster.

This is a contract test, not a perf benchmark; the wall-time numbers
serve as a sanity check that the optimisation took effect.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from ac_load_flow_problem import ACLoadFlowProblemLoader  # noqa: E402

RESULTS_DIR = HERE / "results"

NETWORK = "ieee14"
DATASET_SIZE = 32
BATCH_SIZE = 4
SEED = 0
PERTURBATION_SCALE = 0.1


def _hash_batch(batch) -> str:
    """SHA-256 over all numeric leaves of a ProblemBatch (deterministic order)."""
    h = hashlib.sha256()
    ctx, _ = batch.get_context()
    ora, _ = batch.get_oracle()
    for tree in (ctx, ora):
        for leaf in jax.tree_util.tree_leaves(tree):
            h.update(np.asarray(leaf).tobytes())
    return h.hexdigest()


def _hash_full_iteration(loader: ACLoadFlowProblemLoader) -> list[str]:
    """Hash every batch produced by one iteration of `loader`."""
    return [_hash_batch(b) for b in loader]


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"== Validating ACLoadFlowProblemLoader pre-cache ({NETWORK}) ==")
    print(f"   dataset_size={DATASET_SIZE} batch_size={BATCH_SIZE} seed={SEED}")
    print()

    # Property (1) + (3): build once, iterate twice; first iteration after
    # __init__ should be cheap (cache built during __init__), second iteration
    # likewise. Measure both.
    t0 = time.perf_counter()
    loader_a = ACLoadFlowProblemLoader(
        network_name=NETWORK,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        perturbation_scale=PERTURBATION_SCALE,
    )
    init_time = time.perf_counter() - t0
    print(f"  __init__ (cache build): {init_time:.2f}s")

    t1 = time.perf_counter()
    hashes_epoch_1 = _hash_full_iteration(loader_a)
    epoch_1_time = time.perf_counter() - t1
    print(f"  epoch 1 iteration:      {epoch_1_time:.3f}s  ({len(hashes_epoch_1)} batches)")

    t2 = time.perf_counter()
    hashes_epoch_2 = _hash_full_iteration(loader_a)
    epoch_2_time = time.perf_counter() - t2
    print(f"  epoch 2 iteration:      {epoch_2_time:.3f}s  ({len(hashes_epoch_2)} batches)")

    if hashes_epoch_1 != hashes_epoch_2:
        print("  FAIL: re-iteration produced different batches.")
        return 1
    print("  re-iteration: BYTE-IDENTICAL across two epochs.")

    # Property (2): a fresh loader with the same config produces the same
    # cached instances. This guards against accidental coupling of cache
    # contents to construction call order.
    print()
    print("  Verifying determinism across loader instances...")
    loader_b = ACLoadFlowProblemLoader(
        network_name=NETWORK,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        perturbation_scale=PERTURBATION_SCALE,
    )
    hashes_b = _hash_full_iteration(loader_b)
    if hashes_b != hashes_epoch_1:
        print("  FAIL: two loaders with same config produced different batches.")
        for i, (ha, hb) in enumerate(zip(hashes_epoch_1, hashes_b)):
            if ha != hb:
                print(f"    batch {i}: a={ha[:16]} != b={hb[:16]}")
                break
        return 1
    print("  cross-instance: BYTE-IDENTICAL.")

    # Property (2b): a different seed must produce different batches
    # (otherwise the seed parameter is dead).
    print()
    print("  Verifying seed sensitivity...")
    loader_c = ACLoadFlowProblemLoader(
        network_name=NETWORK,
        dataset_size=DATASET_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED + 1,
        perturbation_scale=PERTURBATION_SCALE,
    )
    hashes_c = _hash_full_iteration(loader_c)
    if hashes_c == hashes_epoch_1:
        print("  FAIL: different seeds produced identical batches (seed is dead).")
        return 1
    print(f"  seed sensitivity: OK (different batches under seed {SEED} vs {SEED + 1}).")

    speedup = epoch_1_time / max(epoch_2_time, 1e-6)
    print()
    print("== Summary ==")
    print(f"  __init__ (1-time cache build): {init_time:.2f}s")
    print(f"  per-epoch iteration time:      {epoch_2_time:.3f}s")
    print(f"  epoch-1 vs epoch-2 speedup:    x{speedup:.1f} (both should be fast post-init)")

    payload = {
        "network": NETWORK,
        "dataset_size": DATASET_SIZE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "perturbation_scale": PERTURBATION_SCALE,
        "init_time_s": init_time,
        "epoch_1_time_s": epoch_1_time,
        "epoch_2_time_s": epoch_2_time,
        "epoch1_epoch2_speedup": speedup,
        "n_batches": len(hashes_epoch_1),
        "batch_hashes_sha256": hashes_epoch_1,
        "checks_passed": ["re_iteration_stable", "cross_instance_deterministic", "seed_sensitive"],
    }
    out_path = RESULTS_DIR / "validate_loader_cache.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nResult written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

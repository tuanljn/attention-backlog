# Copyright (c) 2026, RTE (http://www.rte-france.com)
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
"""
Supervised AC load flow problem on top of pypowsybl-to-energnn.

For a fixed IEEE network topology, generates a stream of supervised
instances by perturbing load setpoints (p0, q0) and solving AC load
flow to obtain ground-truth bus voltages, line flows, generator
outputs, etc. The decision graph predicted by a GNN must match the
load-flow solution under MSE.

This is the Item 0 "data wrapper" prerequisite for Gate 5 in
attention-backlog.md, and is what makes a meaningful baseline /
attention comparison possible on realistic grid topologies.

Topology is fixed per loader instance, so all batches collate cleanly
without padding. Perturbations that fail to converge are retried with
fresh random factors (bounded by max_attempts).
"""
from __future__ import annotations

from copy import deepcopy
from typing import Iterator

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pypowsybl.loadflow as lf
import pypowsybl.network as pn

from energnn.graph import GraphStructure, JaxGraph, collate_graphs_jax
from energnn.problem import Problem, ProblemBatch, ProblemLoader

from pypowsybl_to_energnn.ready_to_use import (
    ACLoadFlowInputConverter,
    ACLoadFlowOutputConverter,
)


def _solve_and_convert(
    network_factory,
    load_factors_p: np.ndarray | None,
    load_factors_q: np.ndarray | None,
    base_loads: pd.DataFrame,
    input_converter: ACLoadFlowInputConverter,
    output_converter: ACLoadFlowOutputConverter,
) -> tuple[JaxGraph, JaxGraph] | None:
    """Build a fresh network, perturb loads, solve AC LF, convert to JaxGraph.

    Returns (context, oracle) on success, or None if load flow did not
    converge. A fresh network is rebuilt every call because pypowsybl
    networks are stateful and re-running load flow on an already-solved
    network with perturbed loads can produce inconsistent state.
    """
    network = network_factory()

    if load_factors_p is not None and len(base_loads) > 0:
        update = pd.DataFrame(
            {
                "p0": base_loads["p0"].to_numpy() * load_factors_p,
                "q0": base_loads["q0"].to_numpy() * load_factors_q,
            },
            index=base_loads.index,
        )
        network.update_loads(update)

    results = lf.run_ac(network)
    if not results:
        return None
    # `status` is the canonical enum; "CONVERGED" is the success state.
    status_name = getattr(results[0].status, "name", str(results[0].status))
    if status_name != "CONVERGED":
        return None

    network.per_unit = True
    context = input_converter(network)
    oracle = output_converter(network)
    return context, oracle


class ACLoadFlowProblem(Problem):
    """A single supervised AC-load-flow instance for one IEEE network state."""

    def __init__(
        self,
        *,
        context: JaxGraph,
        oracle: JaxGraph,
        context_structure: GraphStructure,
        decision_structure: GraphStructure,
    ):
        self._context = context
        self._oracle = oracle
        self._context_structure = context_structure
        self._decision_structure = decision_structure

    @property
    def context_structure(self) -> GraphStructure:
        return self._context_structure

    @property
    def decision_structure(self) -> GraphStructure:
        return self._decision_structure

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[JaxGraph, dict]:
        return deepcopy(self._context), {}

    def get_oracle(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        return deepcopy(self._oracle), {}

    def get_gradient(self, *, decision: JaxGraph, get_info: bool = False, step: int | None = None) -> tuple[JaxGraph, dict]:
        """grad_y f = y - y_oracle for MSE objective."""
        grad = deepcopy(decision)
        grad.feature_flat_array = grad.feature_flat_array - self._oracle.feature_flat_array
        return grad, {}

    def get_score(self, *, decision: JaxGraph, get_info: bool = False, step: int | None = None) -> tuple[float, dict]:
        """f(y;x) = MSE(y, y_oracle), masking NaN to skip fictitious entries."""
        diff = deepcopy(decision)
        diff.feature_flat_array -= self._oracle.feature_flat_array
        score = float(jnp.nanmean(jnp.square(diff.feature_flat_array)))
        return score, {}

    def save(self, *, path: str) -> None:
        # Out of scope for benchmark wrapper.
        pass


class ACLoadFlowProblemBatch(ProblemBatch):
    """A batched AC-load-flow problem with identical topology, varying loads."""

    def __init__(
        self,
        *,
        context_batch: JaxGraph,
        oracle_batch: JaxGraph,
        context_structure: GraphStructure,
        decision_structure: GraphStructure,
    ):
        self._context_batch = context_batch
        self._oracle_batch = oracle_batch
        self._context_structure = context_structure
        self._decision_structure = decision_structure

    @property
    def context_structure(self) -> GraphStructure:
        return self._context_structure

    @property
    def decision_structure(self) -> GraphStructure:
        return self._decision_structure

    def get_context(self, get_info: bool = False, step: int | None = None) -> tuple[JaxGraph, dict]:
        return deepcopy(self._context_batch), {}

    def get_oracle(self, get_info: bool = False) -> tuple[JaxGraph, dict]:
        return deepcopy(self._oracle_batch), {}

    def get_gradient(self, *, decision: JaxGraph, get_info: bool = False, step: int | None = None) -> tuple[JaxGraph, dict]:
        grad = deepcopy(decision)
        grad.feature_flat_array = grad.feature_flat_array - self._oracle_batch.feature_flat_array
        return grad, {}

    def get_score(self, *, decision: JaxGraph, get_info: bool = False, step: int | None = None) -> tuple[list[float], dict]:
        """Per-instance MSE across the batch dimension."""
        diff = deepcopy(decision)
        diff.feature_flat_array -= self._oracle_batch.feature_flat_array
        squared = jnp.nanmean(jnp.square(diff.feature_flat_array), axis=-1)
        return [float(v) for v in np.asarray(squared)], {}


class ACLoadFlowProblemLoader(ProblemLoader):
    """Iterator yielding supervised AC-load-flow batches from a fixed IEEE network.

    Each batch contains :code:`batch_size` problem instances obtained by
    perturbing the network's load setpoints by multiplicative factors
    drawn uniformly from :code:`[1 - perturbation_scale, 1 + perturbation_scale]`.

    **Caching strategy.** The loader pre-generates the full set of
    :code:`dataset_size` instances at construction time (using a private
    RNG seeded with :code:`seed`) and caches the converted
    :code:`(context, decision)` :class:`JaxGraph` pairs. Subsequent
    :code:`__iter__` / :code:`__next__` cycles slice the cache and collate
    — they never re-invoke the pypowsybl AC-LF solver. This is functionally
    identical to the prior on-the-fly behavior (same seed -> same instances)
    but avoids re-solving Newton-Raphson on the CPU at every epoch, which
    was the dominant wall-time cost in training loops. The cache is built
    once and reused across all epochs and (in benchmark scripts that share
    a loader instance) across all attention-variant evaluations, providing
    cleaner cross-variant comparison on identical training data.

    :param network_name: Network identifier (e.g. ``"ieee14"``); resolves
        to ``pypowsybl.network.create_<network_name>``.
    :param dataset_size: Number of supervised instances per pass.
    :param batch_size: Number of instances per yielded batch.
    :param seed: RNG seed for reproducible perturbation sequence.
    :param perturbation_scale: Maximum relative perturbation on (p0, q0).
        ``0.1`` means each load can be scaled by a factor in ``[0.9, 1.1]``.
    :param max_attempts_per_instance: Cap on retries when AC load flow
        fails to converge. With small perturbation_scale this is rarely
        triggered.
    """

    def __init__(
        self,
        *,
        network_name: str,
        dataset_size: int = 32,
        batch_size: int = 4,
        seed: int = 0,
        perturbation_scale: float = 0.1,
        max_attempts_per_instance: int = 8,
    ):
        if not network_name.startswith("ieee"):
            raise ValueError(f"network_name should start with 'ieee', got {network_name!r}")
        factory_name = f"create_{network_name}"
        if not hasattr(pn, factory_name):
            raise ValueError(f"pypowsybl has no factory '{factory_name}'")
        self._network_factory = getattr(pn, factory_name)
        self.network_name = network_name
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.perturbation_scale = float(perturbation_scale)
        self.max_attempts_per_instance = int(max_attempts_per_instance)

        self._input_converter = ACLoadFlowInputConverter()
        self._output_converter = ACLoadFlowOutputConverter()

        # Capture baseline load setpoints from the default network. These do
        # not change between iterations and are used to scale perturbations.
        ref = self._network_factory()
        self._base_loads = ref.get_loads()[["p0", "q0"]].copy()
        # Solve the default once so the loader's structure properties match
        # the converters' get_structure(); also serves as a sanity check.
        lf.run_ac(ref)
        ref.per_unit = True
        _ = self._input_converter(ref)
        _ = self._output_converter(ref)

        self._context_structure = self._input_converter.get_structure()
        self._decision_structure = self._output_converter.get_structure()

        self._current_step = 0
        self._rng: np.random.Generator | None = None
        # Pre-generate the full dataset once. The RNG sequence is deterministic
        # from `seed`, so this is bit-identical to the prior on-the-fly
        # behavior (eval scores unchanged for any prior seed).
        self._cached_pairs: list[tuple[JaxGraph, JaxGraph]] = self._build_cache()

    @property
    def context_structure(self) -> GraphStructure:
        return self._context_structure

    @property
    def decision_structure(self) -> GraphStructure:
        return self._decision_structure

    def __iter__(self) -> Iterator[ACLoadFlowProblemBatch]:
        self._current_step = 0
        return self

    def _draw_factors(self) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._base_loads)
        low = 1.0 - self.perturbation_scale
        high = 1.0 + self.perturbation_scale
        # Independent factors for p and q for richer variation; identical
        # factors would only span a 1-D subspace of the (p, q) plane.
        fp = self._rng.uniform(low, high, size=n)
        fq = self._rng.uniform(low, high, size=n)
        return fp, fq

    def _generate_one(self) -> tuple[JaxGraph, JaxGraph]:
        for _attempt in range(self.max_attempts_per_instance):
            fp, fq = self._draw_factors()
            out = _solve_and_convert(
                network_factory=self._network_factory,
                load_factors_p=fp,
                load_factors_q=fq,
                base_loads=self._base_loads,
                input_converter=self._input_converter,
                output_converter=self._output_converter,
            )
            if out is not None:
                return out
        raise RuntimeError(
            f"AC load flow failed to converge for {self.max_attempts_per_instance} "
            f"random perturbations on {self.network_name}. Consider reducing "
            f"perturbation_scale (currently {self.perturbation_scale})."
        )

    def _build_cache(self) -> list[tuple[JaxGraph, JaxGraph]]:
        """Generate the full :code:`dataset_size` instances once, deterministic from seed."""
        self._rng = np.random.default_rng(self.seed)
        pairs: list[tuple[JaxGraph, JaxGraph]] = []
        for _ in range(self.dataset_size):
            pairs.append(self._generate_one())
        return pairs

    def __next__(self) -> ACLoadFlowProblemBatch:
        if self._current_step >= self.dataset_size:
            raise StopIteration

        batch_start = self._current_step
        batch_end = min(batch_start + self.batch_size, self.dataset_size)
        self._current_step = batch_end

        cached_slice = self._cached_pairs[batch_start:batch_end]
        contexts = [pair[0] for pair in cached_slice]
        oracles = [pair[1] for pair in cached_slice]

        context_batch = collate_graphs_jax(contexts)
        oracle_batch = collate_graphs_jax(oracles)
        return ACLoadFlowProblemBatch(
            context_batch=context_batch,
            oracle_batch=oracle_batch,
            context_structure=self._context_structure,
            decision_structure=self._decision_structure,
        )

    def __len__(self) -> int:
        return max(self.dataset_size // self.batch_size, 1)

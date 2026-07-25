"""Pluggable optimization problems for the genetic engine.

The engine never touches problem internals. It only speaks the small
:class:`Problem` protocol below, so swapping the store-item allocation task
for, say, a TSP or a knapsack is a matter of implementing five methods.

The built-in :class:`AllocationProblem` reconstructs the store-item allocation
work I shipped at Xpectrum: place N catalog items into M shelf slots to
maximize expected sales, penalizing capacity and category-adjacency
violations. Fitness is fully vectorized over a whole population with numpy.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

# A genome is a 1-D integer numpy array (slot -> item index).
Genome = np.ndarray


@runtime_checkable
class Problem(Protocol):
    """The contract every optimization problem must satisfy.

    Implementations own the genome encoding entirely; the engine treats
    genomes as opaque integer vectors and delegates every semantic operation
    (creation, evaluation, recombination, mutation, rendering) here.
    """

    #: Fixed genome length so the engine can pre-allocate population matrices.
    genome_length: int

    def random_genome(self, rng: np.random.Generator) -> Genome:
        """Return one valid random genome using the supplied RNG."""
        ...

    def fitness(self, population: np.ndarray) -> np.ndarray:
        """Vectorized fitness for a ``[pop, genome_length]`` matrix.

        Returns a ``[pop]`` float array. Higher is better.
        """
        ...

    def crossover(
        self, a: Genome, b: Genome, rng: np.random.Generator
    ) -> tuple[Genome, Genome]:
        """Recombine two parents into two children (still valid genomes)."""
        ...

    def mutate(self, genome: Genome, rate: float, rng: np.random.Generator) -> Genome:
        """Mutate a genome in place-safe fashion, keeping it valid."""
        ...

    def render_spec(self, genome: Genome) -> dict[str, Any]:
        """Problem-specific draw data for the frontend (allocation grid)."""
        ...

    def describe(self) -> dict[str, Any]:
        """Static metadata so the UI can build its controls."""
        ...


class AllocationProblem:
    """Store-item allocation: assign items to shelf slots to maximize sales.

    Encoding
    --------
    A genome is a permutation-style assignment of length ``slots`` where
    ``genome[s]`` is the item index placed in slot ``s``. When ``items ==
    slots`` (the default) every genome is a permutation, which is the regime
    the permutation-preserving operators below maintain.

    Fitness
    -------
    ``sum_s affinity[item_s, s] * demand[item_s]``
    ``- capacity_penalty * duplicate_placements``
    ``- adjacency_penalty * category_mismatch_between_neighbors``

    The affinity matrix, per-item demand and per-item category are generated
    deterministically from ``seed`` so a given problem configuration is stable
    across runs and across processes.
    """

    def __init__(
        self,
        items: int = 40,
        slots: int = 40,
        *,
        seed: int = 12345,
        capacity_penalty: float = 25.0,
        adjacency_penalty: float = 4.0,
        categories: int = 5,
    ) -> None:
        if items <= 0 or slots <= 0:
            raise ValueError("items and slots must be positive")
        self.items = items
        self.slots = slots
        self.genome_length = slots
        self.capacity_penalty = capacity_penalty
        self.adjacency_penalty = adjacency_penalty
        self.categories = categories

        # Deterministic problem instance: same config -> same landscape.
        gen = np.random.default_rng(seed)
        # affinity[item, slot] in [0, 1): how well an item performs in a slot.
        self.affinity = gen.random((items, slots))
        # demand[item] in [0.2, 1.2): relative sales volume of each item.
        self.demand = 0.2 + gen.random(items)
        # category[item] in [0, categories): merchandising group of each item.
        self.category = gen.integers(0, categories, size=items)

        # Precompute the per-slot revenue contribution matrix so fitness is a
        # single fancy-indexed gather + sum over the whole population.
        # revenue[item, slot] = affinity[item, slot] * demand[item]
        self.revenue = self.affinity * self.demand[:, None]

    # ------------------------------------------------------------------ #
    # Problem protocol
    # ------------------------------------------------------------------ #
    def random_genome(self, rng: np.random.Generator) -> Genome:
        if self.items == self.slots:
            return rng.permutation(self.items).astype(np.int64)
        # More items than slots: sample without replacement; fewer: allow reuse
        # (duplicates then get penalized by the capacity term).
        replace = self.items < self.slots
        return rng.choice(self.items, size=self.slots, replace=replace).astype(np.int64)

    def fitness(self, population: np.ndarray) -> np.ndarray:
        pop = np.atleast_2d(population)
        n, length = pop.shape
        slot_idx = np.arange(length)

        # Revenue: gather revenue[item_at_slot, slot] for every genome at once.
        revenue = self.revenue[pop, slot_idx].sum(axis=1)

        # Capacity penalty: an item placed in more than one slot violates its
        # single-facing capacity. Count surplus placements per genome.
        # bincount per row via a vectorized histogram over items.
        counts = np.zeros((n, self.items), dtype=np.int64)
        np.add.at(counts, (np.arange(n)[:, None], pop), 1)
        duplicate_placements = np.maximum(counts - 1, 0).sum(axis=1)

        # Adjacency penalty: neighboring slots holding different merchandising
        # categories hurt cross-sell. Count category changes along the shelf.
        cats = self.category[pop]  # [n, length]
        mismatches = (cats[:, 1:] != cats[:, :-1]).sum(axis=1)

        return (
            revenue
            - self.capacity_penalty * duplicate_placements
            - self.adjacency_penalty * mismatches
        ).astype(np.float64)

    def crossover(
        self, a: Genome, b: Genome, rng: np.random.Generator
    ) -> tuple[Genome, Genome]:
        """Order crossover (OX) — preserves permutations when parents are."""
        if self.items != self.slots:
            # Non-permutation regime: uniform crossover is valid and simple.
            mask = rng.random(self.slots) < 0.5
            c1 = np.where(mask, a, b)
            c2 = np.where(mask, b, a)
            return c1.astype(np.int64), c2.astype(np.int64)
        return self._order_crossover(a, b, rng), self._order_crossover(b, a, rng)

    def _order_crossover(
        self, a: Genome, b: Genome, rng: np.random.Generator
    ) -> Genome:
        n = self.slots
        i, j = sorted(rng.integers(0, n, size=2).tolist())
        child = np.full(n, -1, dtype=np.int64)
        child[i : j + 1] = a[i : j + 1]
        taken = set(a[i : j + 1].tolist())
        fill = [x for x in b.tolist() if x not in taken]
        pos = 0
        for k in range(n):
            if child[k] == -1:
                child[k] = fill[pos]
                pos += 1
        return child

    def mutate(self, genome: Genome, rate: float, rng: np.random.Generator) -> Genome:
        out = genome.copy()
        if self.items != self.slots:
            # Point mutation: reassign a slot to a random item.
            flips = rng.random(self.slots) < rate
            out[flips] = rng.integers(0, self.items, size=int(flips.sum()))
            return out
        # Swap mutation: each slot may swap with another, preserving the
        # permutation. Expected number of swaps scales with the rate.
        n = self.slots
        swaps = rng.random(n) < rate
        for s in np.nonzero(swaps)[0]:
            t = int(rng.integers(0, n))
            out[s], out[t] = out[t], out[s]
        return out

    def render_spec(self, genome: Genome) -> dict[str, Any]:
        genome = np.asarray(genome)
        slot_idx = np.arange(self.slots)
        per_slot_value = self.revenue[genome, slot_idx]
        cols = int(np.ceil(np.sqrt(self.slots)))
        rows = int(np.ceil(self.slots / cols))
        cells = [
            {
                "slot": int(s),
                "item": int(genome[s]),
                "row": int(s // cols),
                "col": int(s % cols),
                "category": int(self.category[genome[s]]),
                "value": float(per_slot_value[s]),
            }
            for s in range(self.slots)
        ]
        return {
            "kind": "allocation-grid",
            "rows": rows,
            "cols": cols,
            "categories": self.categories,
            "cells": cells,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "id": "allocation",
            "name": "Store-Item Allocation",
            "description": (
                "Assign catalog items to shelf slots to maximize expected "
                "sales, penalizing duplicate placements and category-"
                "adjacency violations."
            ),
            "genomeLength": self.genome_length,
            "config": {"items": self.items, "slots": self.slots},
            "configSchema": {
                "items": {"type": "int", "min": 4, "max": 400, "default": 40},
                "slots": {"type": "int", "min": 4, "max": 400, "default": 40},
            },
        }


def build_problem(name: str, config: dict[str, Any] | None, seed: int) -> Problem:
    """Factory used by the runner to materialize a problem from RunParams."""
    config = config or {}
    if name == "allocation":
        items = int(config.get("items", 40))
        slots = int(config.get("slots", 40))
        return AllocationProblem(items=items, slots=slots, seed=seed)
    raise ValueError(f"unknown problem: {name!r}")


#: Registry powering ``GET /api/problems``.
PROBLEMS: dict[str, Problem] = {"allocation": AllocationProblem()}

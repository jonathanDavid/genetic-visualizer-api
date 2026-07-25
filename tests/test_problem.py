"""Problem-level invariants: valid genomes and vectorized fitness."""

from __future__ import annotations

import numpy as np
import pytest

from ga.problem import AllocationProblem


@pytest.fixture()
def problem() -> AllocationProblem:
    return AllocationProblem(items=20, slots=20, seed=7)


def _is_permutation(genome: np.ndarray, n: int) -> bool:
    return sorted(genome.tolist()) == list(range(n))


def test_random_genome_is_valid_permutation(problem: AllocationProblem) -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = problem.random_genome(rng)
        assert g.shape == (problem.slots,)
        assert _is_permutation(g, problem.items)


def test_crossover_preserves_permutation(problem: AllocationProblem) -> None:
    rng = np.random.default_rng(1)
    a = problem.random_genome(rng)
    b = problem.random_genome(rng)
    for _ in range(50):
        c1, c2 = problem.crossover(a, b, rng)
        assert _is_permutation(c1, problem.items)
        assert _is_permutation(c2, problem.items)


def test_mutation_preserves_permutation(problem: AllocationProblem) -> None:
    rng = np.random.default_rng(2)
    g = problem.random_genome(rng)
    for _ in range(50):
        m = problem.mutate(g, rate=0.3, rng=rng)
        assert _is_permutation(m, problem.items)
        assert m is not g  # returns a copy, never mutates the input


def test_fitness_vectorized_matches_scalar(problem: AllocationProblem) -> None:
    rng = np.random.default_rng(3)
    pop = np.asarray([problem.random_genome(rng) for _ in range(32)])
    batch = problem.fitness(pop)

    # Independently recompute each genome's fitness via a single-row call.
    scalar = np.array([problem.fitness(pop[i : i + 1])[0] for i in range(len(pop))])
    assert np.allclose(batch, scalar)
    assert batch.shape == (32,)


def test_fitness_penalizes_duplicate_placements() -> None:
    problem = AllocationProblem(items=10, slots=10, seed=5)
    clean = np.arange(10)[None, :]
    dupes = np.zeros((1, 10), dtype=np.int64)  # item 0 in every slot
    assert problem.fitness(clean)[0] > problem.fitness(dupes)[0]


def test_render_spec_shape(problem: AllocationProblem) -> None:
    rng = np.random.default_rng(4)
    g = problem.random_genome(rng)
    spec = problem.render_spec(g)
    assert spec["kind"] == "allocation-grid"
    assert len(spec["cells"]) == problem.slots
    assert {"slot", "item", "row", "col", "category", "value"} <= spec["cells"][0].keys()

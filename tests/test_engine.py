"""Engine-level invariants: elitism, reproducibility, selection pressure."""

from __future__ import annotations

import numpy as np
import pytest

from ga.engine import EngineConfig, GeneticEngine
from ga.problem import AllocationProblem


def _engine(seed: int | None, **overrides) -> GeneticEngine:
    problem = AllocationProblem(items=24, slots=24, seed=99)
    params = {
        "population_size": 60,
        "generations": 40,
        "mutation_rate": 0.1,
        "crossover_rate": 0.9,
        "elitism": 2,
        "selection": "tournament",
        "seed": seed,
    }
    params.update(overrides)
    return GeneticEngine(problem, EngineConfig(**params))


def test_best_fitness_non_decreasing_with_elitism() -> None:
    engine = _engine(seed=1, elitism=2)
    best = [r.best_fitness for r in engine.run()]
    for prev, cur in zip(best, best[1:]):
        assert cur >= prev - 1e-9, f"best fitness dropped: {prev} -> {cur}"


def test_seed_reproducibility() -> None:
    a = list(_engine(seed=123).run())
    b = list(_engine(seed=123).run())
    assert len(a) == len(b)
    for ra, rb in zip(a, b):
        assert ra.best_fitness == rb.best_fitness
        assert np.array_equal(ra.best_genome, rb.best_genome)


def test_different_seeds_diverge() -> None:
    a = list(_engine(seed=1).run())
    b = list(_engine(seed=2).run())
    # Overwhelmingly likely the final best genomes differ across seeds.
    assert not np.array_equal(a[-1].best_genome, b[-1].best_genome)


def test_tournament_selection_favors_fitness() -> None:
    engine = _engine(seed=7)
    # Fitness where index i has fitness i: the max index must be picked far
    # more often than uniform (1/n) would predict.
    n = 60
    fitness = np.arange(n, dtype=float)
    picks = engine._tournament(fitness, 5000)
    top_share = np.mean(picks >= n - 5)
    # With k=3, P(winner in top 5 of 60) ≈ 0.23; uniform would be ~0.083.
    assert top_share > 0.15


def test_roulette_selection_favors_fitness() -> None:
    engine = _engine(seed=7, selection="roulette")
    n = 40
    fitness = np.arange(n, dtype=float)
    picks = engine._roulette(fitness, 5000)
    assert picks.mean() > (n - 1) / 2  # skewed above the uniform mean


def test_final_fitness_improves_over_initial() -> None:
    engine = _engine(seed=3)
    results = list(engine.run())
    assert results[-1].best_fitness > results[0].best_fitness


def test_stop_callback_halts_early() -> None:
    engine = _engine(seed=5)
    seen: list[int] = []

    def stop() -> bool:
        return len(seen) >= 3

    for r in engine.run(should_stop=stop):
        seen.append(r.gen)
    assert len(seen) == 3


def test_diversity_in_unit_range() -> None:
    engine = _engine(seed=9)
    for r in engine.run():
        assert 0.0 <= r.diversity <= 1.0


@pytest.mark.parametrize("selection", ["tournament", "roulette"])
def test_both_selection_strategies_run(selection: str) -> None:
    engine = _engine(seed=11, selection=selection)
    results = list(engine.run())
    assert len(results) == 40

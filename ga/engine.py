"""Seedable genetic-algorithm loop.

The engine is deliberately problem-agnostic: it holds a population of integer
genomes as a numpy matrix and drives selection -> crossover -> mutation ->
elitism each generation, delegating every semantic operation to a
:class:`~ga.problem.Problem`. ``run()`` is a generator that yields one
:class:`GenerationResult` per generation, which the runner turns into
WebSocket events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Literal

import numpy as np

from .problem import Problem

Selection = Literal["tournament", "roulette"]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Per-generation summary emitted by the engine."""

    gen: int
    best_fitness: float
    avg_fitness: float
    worst_fitness: float
    diversity: float
    best_genome: np.ndarray


@dataclass(slots=True)
class EngineConfig:
    population_size: int = 120
    generations: int = 200
    mutation_rate: float = 0.05
    crossover_rate: float = 0.9
    elitism: int = 2
    selection: Selection = "tournament"
    seed: int | None = None
    tournament_k: int = 3


class GeneticEngine:
    """A single-population, elitist genetic algorithm."""

    def __init__(self, problem: Problem, config: EngineConfig) -> None:
        self.problem = problem
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self._population = self._init_population()

    # ------------------------------------------------------------------ #
    # Population lifecycle
    # ------------------------------------------------------------------ #
    def _init_population(self) -> np.ndarray:
        rows = [self.problem.random_genome(self.rng) for _ in range(self.cfg.population_size)]
        return np.asarray(rows, dtype=np.int64)

    # ------------------------------------------------------------------ #
    # Selection strategies
    # ------------------------------------------------------------------ #
    def _select_parents(self, fitness: np.ndarray, n: int) -> np.ndarray:
        """Return ``n`` parent indices according to the configured strategy."""
        if self.cfg.selection == "roulette":
            return self._roulette(fitness, n)
        return self._tournament(fitness, n)

    def _tournament(self, fitness: np.ndarray, n: int) -> np.ndarray:
        pop = fitness.shape[0]
        k = min(self.cfg.tournament_k, pop)
        # [n, k] contenders; winner is the fittest in each row.
        contenders = self.rng.integers(0, pop, size=(n, k))
        contender_fitness = fitness[contenders]
        winners = contenders[np.arange(n), contender_fitness.argmax(axis=1)]
        return winners

    def _roulette(self, fitness: np.ndarray, n: int) -> np.ndarray:
        # Shift so the minimum contributes a small non-zero weight even when
        # fitness is negative (allocation fitness can dip below zero).
        weights = fitness - fitness.min()
        total = weights.sum()
        if total <= 0:
            probs = np.full(fitness.shape[0], 1.0 / fitness.shape[0])
        else:
            probs = weights / total
        return self.rng.choice(fitness.shape[0], size=n, p=probs)

    # ------------------------------------------------------------------ #
    # Diversity
    # ------------------------------------------------------------------ #
    def _diversity(self, population: np.ndarray) -> float:
        """Normalized genome spread in ``[0, 1]``.

        Uses the expected per-position disagreement between two random
        genomes: ``mean_columns(1 - sum_v p_v^2)``. This equals the expected
        normalized Hamming distance and costs O(pop * length) rather than the
        O(pop^2 * length) of explicit pairwise distances.
        """
        n, length = population.shape
        if n < 2:
            return 0.0
        acc = 0.0
        for col in range(length):
            counts = np.bincount(population[:, col])
            p = counts / n
            acc += 1.0 - float((p * p).sum())
        return acc / length

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, should_stop: Callable[[], bool] | None = None) -> Iterator[GenerationResult]:
        """Yield a :class:`GenerationResult` per generation.

        ``should_stop`` is polled once per generation; when it returns True the
        loop halts early (used by the run registry's stop endpoint).
        """
        cfg = self.cfg
        pop = self._population

        for gen in range(cfg.generations):
            fitness = self.problem.fitness(pop)
            order = fitness.argsort()[::-1]  # descending by fitness
            best_idx = int(order[0])

            yield GenerationResult(
                gen=gen,
                best_fitness=float(fitness[best_idx]),
                avg_fitness=float(fitness.mean()),
                worst_fitness=float(fitness.min()),
                diversity=self._diversity(pop),
                best_genome=pop[best_idx].copy(),
            )

            if should_stop is not None and should_stop():
                self._population = pop
                return

            pop = self._next_generation(pop, fitness, order)

        self._population = pop

    def _next_generation(
        self, pop: np.ndarray, fitness: np.ndarray, order: np.ndarray
    ) -> np.ndarray:
        cfg = self.cfg
        size = cfg.population_size
        elite = min(cfg.elitism, size)

        children: list[np.ndarray] = []
        # Elitism: carry the top genomes forward untouched. This is what makes
        # best-fitness non-decreasing across generations.
        for i in range(elite):
            children.append(pop[order[i]].copy())

        needed = size - elite
        # Draw all parents up front for the non-elite slots.
        parents = self._select_parents(fitness, 2 * needed)

        i = 0
        while len(children) < size:
            pa = pop[parents[i % len(parents)]]
            pb = pop[parents[(i + 1) % len(parents)]]
            i += 2
            if self.rng.random() < cfg.crossover_rate:
                c1, c2 = self.problem.crossover(pa, pb, self.rng)
            else:
                c1, c2 = pa.copy(), pb.copy()
            c1 = self.problem.mutate(c1, cfg.mutation_rate, self.rng)
            children.append(c1)
            if len(children) < size:
                c2 = self.problem.mutate(c2, cfg.mutation_rate, self.rng)
                children.append(c2)

        return np.asarray(children, dtype=np.int64)

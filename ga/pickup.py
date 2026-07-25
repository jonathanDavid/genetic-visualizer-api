"""PickupProblem: which store supplies each item on a shopping list.

Genome: one gene per needed product — the index of the store that supplies
it. Only stores that stock the product are valid; foreign genes are repaired
by mapping them into the item's valid-store list (``gene % len(valid)``), so
every genome the engine can produce evaluates to a meaningful plan.

Fitness (maximize):

    -( routeKm * KM_COST  +  itemCostCents * CENTS_SCALE  +  storesUsed * STOP_PENALTY )

where the route is customer → selected stores in nearest-neighbor order →
customer. Fewer stops, shorter routes and cheaper baskets all raise fitness,
and the three terms are scaled to compete so the GA has real trade-offs to
discover (drive further for a cheaper price? consolidate into one stop?).

Implements the same structural :class:`~ga.problem.Problem` protocol as
AllocationProblem, so the engine and runner need zero changes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .scenario import Scenario, generate_scenario, travel_minutes

# Cost weights, in "fitness points". A typical 6-store scenario yields routes
# of 5-30 km and baskets of a few million COP cents, so with these weights all
# three terms land in the same order of magnitude.
KM_COST = 40.0          # points per km driven
STOP_PENALTY = 150.0    # points per distinct store visited
CENTS_SCALE = 1e-4      # points per COP cent (1 point per 100 COP)


class PickupProblem:
    """Grocery pickup planning over a deterministic :class:`Scenario`."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        km_cost: float = KM_COST,
        stop_penalty: float = STOP_PENALTY,
        cents_scale: float = CENTS_SCALE,
    ) -> None:
        self.scenario = scenario
        self.km_cost = km_cost
        self.stop_penalty = stop_penalty
        self.cents_scale = cents_scale

        needs = scenario.shopping_list
        self.genome_length = len(needs)
        self._skus = [n.sku for n in needs]

        # valid_stores[i]: store indices stocking needed item i (never empty —
        # the generator guarantees coverage).
        self._valid_stores: list[np.ndarray] = []
        for n in needs:
            stocking = scenario.stores_stocking(n.sku)
            if not stocking:
                raise ValueError(f"shopping-list item {n.sku} is not stocked anywhere")
            self._valid_stores.append(np.asarray(stocking, dtype=np.int64))

        n_stores = len(scenario.stores)
        # price[store, item] in cents; +inf where the store lacks the item
        # (never read after repair, but keeps mistakes loud).
        self._price = np.full((n_stores, self.genome_length), np.inf)
        self._valid_mask = np.zeros((n_stores, self.genome_length), dtype=bool)
        for i, n in enumerate(needs):
            for s in self._valid_stores[i].tolist():
                self._price[s, i] = scenario.price_cents(s, n.sku)
                self._valid_mask[s, i] = True

        self._coords = np.asarray([[s.x, s.y] for s in scenario.stores])
        self._route_cache: dict[frozenset[int], tuple[list[int], float]] = {}

    # ------------------------------------------------------------------ #
    # Genome validity
    # ------------------------------------------------------------------ #
    def repair(self, population: np.ndarray) -> np.ndarray:
        """Clamp invalid genes into each item's valid-store list."""
        pop = np.atleast_2d(population).astype(np.int64, copy=True)
        n_stores = len(self.scenario.stores)
        for i in range(self.genome_length):
            col = pop[:, i]
            valid = self._valid_stores[i]
            in_range = (col >= 0) & (col < n_stores)
            ok = np.zeros(col.shape, dtype=bool)
            ok[in_range] = self._valid_mask[col[in_range], i]
            bad = ~ok
            if bad.any():
                pop[bad, i] = valid[np.abs(col[bad]) % valid.size]
        return pop

    # ------------------------------------------------------------------ #
    # Problem protocol
    # ------------------------------------------------------------------ #
    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(
            [valid[rng.integers(0, valid.size)] for valid in self._valid_stores],
            dtype=np.int64,
        )

    def fitness(self, population: np.ndarray) -> np.ndarray:
        pop = self.repair(population)
        n = pop.shape[0]
        item_idx = np.arange(self.genome_length)

        # Basket cost: vectorized gather over the whole population.
        cost_cents = self._price[pop, item_idx].sum(axis=1)

        out = np.empty(n, dtype=np.float64)
        for r in range(n):
            stores_used = frozenset(pop[r].tolist())
            _, route_km = self._route(stores_used)
            out[r] = -(
                route_km * self.km_cost
                + cost_cents[r] * self.cents_scale
                + len(stores_used) * self.stop_penalty
            )
        return out

    def crossover(
        self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Uniform per-item crossover — children inherit valid genes only."""
        mask = rng.random(self.genome_length) < 0.5
        c1 = np.where(mask, a, b).astype(np.int64)
        c2 = np.where(mask, b, a).astype(np.int64)
        return c1, c2

    def mutate(self, genome: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
        """Reassign each item, with probability ``rate``, to another valid store."""
        out = genome.copy()
        for i in range(self.genome_length):
            if rng.random() >= rate:
                continue
            valid = self._valid_stores[i]
            if valid.size == 1:
                out[i] = valid[0]
                continue
            alternatives = valid[valid != out[i]]
            pool = alternatives if alternatives.size else valid
            out[i] = pool[rng.integers(0, pool.size)]
        return out

    def render_spec(self, genome: np.ndarray) -> dict[str, Any]:
        genome = self.repair(genome)[0]
        route_order, route_km = self._route(frozenset(genome.tolist()))
        item_idx = np.arange(self.genome_length)
        cost_cents = int(self._price[genome, item_idx].sum())
        return {
            "selection": {
                self._skus[i]: self.scenario.stores[int(genome[i])].id
                for i in range(self.genome_length)
            },
            "route": [self.scenario.stores[s].id for s in route_order],
            "routeKm": round(route_km, 3),
            "itemCostCents": cost_cents,
            "storesUsed": len(set(genome.tolist())),
            "travelMinutes": round(travel_minutes(route_km), 2),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "id": "pickup",
            "name": "Grocery Pickup Route",
            "description": (
                "Choose which store supplies each item on a shopping list, "
                "balancing route length, basket price and number of stops."
            ),
            "genomeLength": self.genome_length,
            "config": {
                "seed": self.scenario.seed,
                "stores": len(self.scenario.stores),
                "products": None,
                "needs": len(self.scenario.shopping_list),
            },
            "configSchema": {
                "seed": {"type": "int", "min": 0, "max": 10_000_000, "default": 42},
                "stores": {"type": "int", "min": 3, "max": 12, "default": 6},
                "products": {"type": "int", "min": 6, "max": 30, "default": 12},
                "needs": {"type": "int", "min": 3, "max": 10, "default": 5},
            },
        }

    # ------------------------------------------------------------------ #
    # Metrics / baselines
    # ------------------------------------------------------------------ #
    def metrics(self, genome: np.ndarray) -> dict[str, Any]:
        """Baseline-shaped metrics for a single genome (contract shape)."""
        spec = self.render_spec(np.asarray(genome))
        fitness = float(self.fitness(np.asarray(genome)[None, :])[0])
        return {
            "routeKm": spec["routeKm"],
            "itemCostCents": spec["itemCostCents"],
            "storesUsed": spec["storesUsed"],
            "travelMinutes": spec["travelMinutes"],
            "fitness": round(fitness, 4),
        }

    def greedy_genome(self) -> np.ndarray:
        """Each item from the nearest store stocking it (contract greedy)."""
        distances = np.asarray([s.distance_km for s in self.scenario.stores])
        return np.asarray(
            [valid[np.argmin(distances[valid])] for valid in self._valid_stores],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------ #
    # Routing
    # ------------------------------------------------------------------ #
    def _route(self, stores_used: frozenset[int]) -> tuple[list[int], float]:
        """Nearest-neighbor tour customer → stores → customer (memoized)."""
        cached = self._route_cache.get(stores_used)
        if cached is not None:
            return cached
        remaining = set(stores_used)
        pos = np.zeros(2)
        order: list[int] = []
        km = 0.0
        while remaining:
            idx_list = sorted(remaining)
            dists = np.linalg.norm(self._coords[idx_list] - pos, axis=1)
            pick = idx_list[int(np.argmin(dists))]
            km += float(dists.min())
            pos = self._coords[pick]
            order.append(pick)
            remaining.remove(pick)
        km += float(np.linalg.norm(pos))  # back to the customer at the origin
        result = (order, km)
        self._route_cache[stores_used] = result
        return result


def build_pickup_problem(config: dict[str, Any] | None, fallback_seed: int) -> PickupProblem:
    """Materialize a PickupProblem from RunParams' ``problemConfig``."""
    config = config or {}
    scenario = generate_scenario(
        seed=int(config.get("seed", fallback_seed)),
        stores=int(config.get("stores", 6)),
        products=int(config.get("products", 12)),
        needs=int(config.get("needs", 5)),
    )
    return PickupProblem(scenario)

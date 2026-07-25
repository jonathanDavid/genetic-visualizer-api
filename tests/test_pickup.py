"""v2 pickup problem: scenario determinism, genome validity, fitness shape."""

from __future__ import annotations

import numpy as np
import pytest

from ga.pickup import PickupProblem
from ga.scenario import (
    InventoryItem,
    NeededProduct,
    Scenario,
    Store,
    generate_scenario,
)


# --------------------------------------------------------------------- #
# Scenario generator
# --------------------------------------------------------------------- #
def test_scenario_determinism_same_seed_identical() -> None:
    a = generate_scenario(seed=7, stores=6, products=12, needs=5).to_dict()
    b = generate_scenario(seed=7, stores=6, products=12, needs=5).to_dict()
    assert a == b


def test_scenario_different_seeds_differ() -> None:
    a = generate_scenario(seed=1).to_dict()
    b = generate_scenario(seed=2).to_dict()
    assert a != b


@pytest.mark.parametrize("seed", range(20))
def test_shopping_list_always_coverable(seed: int) -> None:
    sc = generate_scenario(seed=seed, stores=3, products=30, needs=10)
    for needed in sc.shopping_list:
        assert sc.stores_stocking(needed.sku), f"{needed.sku} not stocked anywhere"


def test_scenario_contract_shape() -> None:
    d = generate_scenario(seed=3, stores=4, products=8, needs=4).to_dict()
    assert d["seed"] == 3
    assert d["customer"] == {"x": 0.0, "y": 0.0}
    assert len(d["stores"]) == 4
    store = d["stores"][0]
    assert {"id", "name", "x", "y", "distanceKm", "inventory"} <= store.keys()
    assert {"sku", "name", "priceCents", "stock"} <= store["inventory"][0].keys()
    assert len(d["shoppingList"]) == 4
    assert {"sku", "name"} <= d["shoppingList"][0].keys()
    # Prices are integer COP cents with per-store variation around the base.
    assert all(
        isinstance(it["priceCents"], int) and it["priceCents"] > 0
        for s in d["stores"]
        for it in s["inventory"]
    )


# --------------------------------------------------------------------- #
# PickupProblem genome validity
# --------------------------------------------------------------------- #
@pytest.fixture()
def problem() -> PickupProblem:
    return PickupProblem(generate_scenario(seed=11, stores=6, products=12, needs=5))


def _is_valid(problem: PickupProblem, genome: np.ndarray) -> bool:
    sc = problem.scenario
    return all(
        int(genome[i]) in sc.stores_stocking(need.sku)
        for i, need in enumerate(sc.shopping_list)
    )


def test_random_genome_valid(problem: PickupProblem) -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        g = problem.random_genome(rng)
        assert g.shape == (problem.genome_length,)
        assert _is_valid(problem, g)


def test_crossover_and_mutation_stay_valid(problem: PickupProblem) -> None:
    rng = np.random.default_rng(1)
    a = problem.random_genome(rng)
    b = problem.random_genome(rng)
    for _ in range(50):
        c1, c2 = problem.crossover(a, b, rng)
        assert _is_valid(problem, c1)
        assert _is_valid(problem, c2)
        m = problem.mutate(c1, rate=0.5, rng=rng)
        assert _is_valid(problem, m)
        a, b = m, c2


def test_repair_clamps_invalid_genes(problem: PickupProblem) -> None:
    bogus = np.full((1, problem.genome_length), 999, dtype=np.int64)
    repaired = problem.repair(bogus)[0]
    assert _is_valid(problem, repaired)
    assert np.isfinite(problem.fitness(bogus)).all()


# --------------------------------------------------------------------- #
# Fitness semantics on a crafted scenario
# --------------------------------------------------------------------- #
def _crafted_scenario() -> Scenario:
    """Two stores, identical prices and stock: A near (1 km), B far (12 km)."""
    needs = [NeededProduct(sku=f"p{i}", name=f"Item {i}") for i in range(3)]

    def inventory() -> list[InventoryItem]:
        return [
            InventoryItem(sku=n.sku, name=n.name, price_cents=100_000, stock=10)
            for n in needs
        ]

    stores = [
        Store(id="s1", name="Near", x=1.0, y=0.0, distance_km=1.0, inventory=inventory()),
        Store(id="s2", name="Far", x=12.0, y=0.0, distance_km=12.0, inventory=inventory()),
    ]
    return Scenario(seed=0, customer=(0.0, 0.0), stores=stores, shopping_list=needs)


def test_fitness_prefers_fewer_stops() -> None:
    problem = PickupProblem(_crafted_scenario())
    one_stop = np.array([[0, 0, 0]])
    two_stops = np.array([[0, 0, 1]])  # same prices, extra store + longer route
    assert problem.fitness(one_stop)[0] > problem.fitness(two_stops)[0]


def test_fitness_prefers_shorter_route() -> None:
    problem = PickupProblem(_crafted_scenario())
    near_only = np.array([[0, 0, 0]])
    far_only = np.array([[1, 1, 1]])  # one stop as well, but 24 km round trip
    assert problem.fitness(near_only)[0] > problem.fitness(far_only)[0]


def test_render_spec_contract_shape(problem: PickupProblem) -> None:
    rng = np.random.default_rng(4)
    spec = problem.render_spec(problem.random_genome(rng))
    assert {"selection", "route", "routeKm", "itemCostCents",
            "storesUsed", "travelMinutes"} <= spec.keys()
    assert set(spec["selection"].keys()) == {n.sku for n in problem.scenario.shopping_list}
    store_ids = {s.id for s in problem.scenario.stores}
    assert set(spec["selection"].values()) <= store_ids
    assert set(spec["route"]) <= store_ids
    assert len(spec["route"]) == spec["storesUsed"]
    assert spec["travelMinutes"] == pytest.approx(spec["routeKm"] / 30 * 60, abs=0.02)


# --------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------- #
def test_greedy_beats_random_on_average() -> None:
    greedy_scores: list[float] = []
    random_scores: list[float] = []
    for seed in range(20):
        p = PickupProblem(generate_scenario(seed=seed))
        rng = np.random.default_rng(seed)
        random_scores.append(p.metrics(p.random_genome(rng))["fitness"])
        greedy_scores.append(p.metrics(p.greedy_genome())["fitness"])
    assert np.mean(greedy_scores) > np.mean(random_scores)


def test_greedy_uses_nearest_stocking_store(problem: PickupProblem) -> None:
    sc = problem.scenario
    genome = problem.greedy_genome()
    for i, need in enumerate(sc.shopping_list):
        valid = sc.stores_stocking(need.sku)
        nearest = min(valid, key=lambda s: sc.stores[s].distance_km)
        assert int(genome[i]) == nearest

"""Custom (explicit) scenario for the retail integration.

Covers: explicit scenario used verbatim, coverage validation (422 on an
uncoverable SKU), qty participation in itemCostCents, greedy consolidation
beating a scattered per-item-nearest plan, and the POST baselines surface.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from ga.pickup import PickupProblem, build_pickup_problem
from ga.scenario import ScenarioValidationError, scenario_from_dict

client = TestClient(app)


def _item(sku: str, price: int, stock: int = 10, name: str | None = None) -> dict:
    d = {"sku": sku, "priceCents": price, "stock": stock}
    if name is not None:
        d["name"] = name
    return d


def _scenario() -> dict:
    return {
        "customer": {"x": 0.0, "y": 0.0},
        "stores": [
            {
                "id": "s1",
                "name": "Tienda El Prado",
                "x": 2.0,
                "y": -1.0,
                "inventory": [_item("ARROZ-500", 345_000), _item("LECHE-1L", 480_000)],
            },
            {
                "id": "s2",
                "name": "Super Costa",
                "x": -3.0,
                "y": 4.0,
                "inventory": [_item("LECHE-1L", 500_000), _item("CAFE-250", 1_200_000)],
            },
        ],
        "shoppingList": [{"sku": "ARROZ-500", "qty": 2}, {"sku": "LECHE-1L", "qty": 1}],
    }


# --------------------------------------------------------------------- #
# Verbatim use + determinism
# --------------------------------------------------------------------- #
def test_explicit_scenario_used_verbatim() -> None:
    sc = scenario_from_dict(_scenario())
    assert [s.id for s in sc.stores] == ["s1", "s2"]
    assert sc.stores[0].name == "Tienda El Prado"
    assert sc.stores[0].x == 2.0 and sc.stores[0].y == -1.0
    # distanceKm derived from the customer (origin here).
    assert sc.stores[0].distance_km == pytest.approx(np.hypot(2.0, 1.0), abs=1e-3)
    assert [n.sku for n in sc.shopping_list] == ["ARROZ-500", "LECHE-1L"]
    assert sc.shopping_list[0].qty == 2


def test_explicit_scenario_same_in_same_plan() -> None:
    p1 = PickupProblem(scenario_from_dict(_scenario()))
    p2 = PickupProblem(scenario_from_dict(_scenario()))
    g = p1.greedy_genome()
    assert p1.render_spec(g) == p2.render_spec(g)
    assert p1.metrics(g) == p2.metrics(g)


def test_build_pickup_problem_prefers_explicit_scenario() -> None:
    problem = build_pickup_problem({"scenario": _scenario(), "seed": 999}, fallback_seed=1)
    # SKUs come from the explicit order, not the seeded catalog.
    assert set(problem._skus) == {"ARROZ-500", "LECHE-1L"}


# --------------------------------------------------------------------- #
# Coverage validation
# --------------------------------------------------------------------- #
def test_uncoverable_sku_rejected() -> None:
    bad = _scenario()
    bad["shoppingList"].append({"sku": "PAN-NOEXISTE", "qty": 1})
    with pytest.raises(ScenarioValidationError) as exc:
        scenario_from_dict(bad)
    assert "PAN-NOEXISTE" in str(exc.value)


def test_run_with_uncoverable_scenario_returns_422() -> None:
    bad = _scenario()
    bad["shoppingList"].append({"sku": "GHOST-1", "qty": 1})
    resp = client.post(
        "/api/runs",
        json={"problem": "pickup", "generations": 12, "problemConfig": {"scenario": bad}},
    )
    assert resp.status_code == 422
    assert "GHOST-1" in resp.json()["detail"]


def test_malformed_scenario_rejected() -> None:
    with pytest.raises(ScenarioValidationError):
        scenario_from_dict({"stores": [], "shoppingList": [{"sku": "X"}]})


# --------------------------------------------------------------------- #
# qty affects item cost
# --------------------------------------------------------------------- #
def test_qty_scales_item_cost() -> None:
    one = _scenario()
    one["shoppingList"] = [{"sku": "ARROZ-500", "qty": 1}]
    three = _scenario()
    three["shoppingList"] = [{"sku": "ARROZ-500", "qty": 3}]

    p1 = PickupProblem(scenario_from_dict(one))
    p3 = PickupProblem(scenario_from_dict(three))
    g1 = p1.greedy_genome()
    g3 = p3.greedy_genome()
    # Same store/price, tripled quantity -> triple the item cost.
    assert p3.render_spec(g3)["itemCostCents"] == 3 * p1.render_spec(g1)["itemCostCents"]


# --------------------------------------------------------------------- #
# Greedy consolidation beats a scattered per-item-nearest plan
# --------------------------------------------------------------------- #
def _headroom_scenario() -> dict:
    """One store stocks everything cheaply; each item also sits in its own
    far-flung store. A per-item-nearest plan scatters; consolidating into the
    single hub store is shorter and cheaper."""
    hub = {
        "id": "hub",
        "name": "Hub",
        "x": 1.0,
        "y": 0.0,
        "inventory": [_item("A", 100_000), _item("B", 100_000), _item("C", 100_000)],
    }
    far_a = {"id": "fa", "name": "FarA", "x": 0.4, "y": 0.0,
             "inventory": [_item("A", 100_000)]}
    far_b = {"id": "fb", "name": "FarB", "x": -6.0, "y": 6.0,
             "inventory": [_item("B", 100_000)]}
    far_c = {"id": "fc", "name": "FarC", "x": 7.0, "y": -7.0,
             "inventory": [_item("C", 100_000)]}
    return {
        "customer": {"x": 0.0, "y": 0.0},
        "stores": [hub, far_a, far_b, far_c],
        "shoppingList": [{"sku": "A", "qty": 1}, {"sku": "B", "qty": 1}, {"sku": "C", "qty": 1}],
    }


def test_consolidated_plan_beats_scattered_nearest() -> None:
    problem = PickupProblem(scenario_from_dict(_headroom_scenario()))
    ids = [s.id for s in problem.scenario.stores]
    hub = ids.index("hub")

    consolidated = np.array([[hub, hub, hub]])
    # Scattered per-item-nearest: A from the marginally-closer FarA, B/C from
    # their only far stores.
    scattered = np.array([[ids.index("fa"), ids.index("fb"), ids.index("fc")]])
    assert problem.fitness(consolidated)[0] > problem.fitness(scattered)[0]


# --------------------------------------------------------------------- #
# POST /api/scenario/baselines
# --------------------------------------------------------------------- #
def test_post_baselines_shape_and_determinism() -> None:
    resp = client.post("/api/scenario/baselines", json={"scenario": _headroom_scenario()})
    assert resp.status_code == 200
    body = resp.json()
    keys = {"routeKm", "itemCostCents", "storesUsed", "travelMinutes", "fitness"}
    for kind in ("random", "greedy"):
        assert keys <= body[kind].keys()
    # Same explicit scenario -> identical baselines.
    again = client.post("/api/scenario/baselines", json={"scenario": _headroom_scenario()})
    assert again.json() == body


def test_post_baselines_accepts_bare_scenario_body() -> None:
    resp = client.post("/api/scenario/baselines", json=_scenario())
    assert resp.status_code == 200
    assert "greedy" in resp.json()


def test_post_baselines_uncoverable_422() -> None:
    bad = _scenario()
    bad["shoppingList"].append({"sku": "NOPE", "qty": 1})
    resp = client.post("/api/scenario/baselines", json={"scenario": bad})
    assert resp.status_code == 422
    assert "NOPE" in resp.json()["detail"]

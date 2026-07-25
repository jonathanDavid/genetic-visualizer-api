"""REST surface for the v2 pickup scenario: /api/scenario, baselines, runs."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_scenario_endpoint_shape_and_determinism() -> None:
    a = client.get("/api/scenario", params={"seed": 5, "stores": 4, "products": 8, "needs": 3})
    b = client.get("/api/scenario", params={"seed": 5, "stores": 4, "products": 8, "needs": 3})
    assert a.status_code == 200
    assert a.json() == b.json()
    body = a.json()
    assert body["seed"] == 5
    assert len(body["stores"]) == 4
    assert len(body["shoppingList"]) == 3
    assert {"id", "name", "x", "y", "distanceKm", "inventory"} <= body["stores"][0].keys()


def test_scenario_endpoint_defaults() -> None:
    body = client.get("/api/scenario").json()
    assert len(body["stores"]) == 6
    assert len(body["shoppingList"]) == 5


def test_scenario_endpoint_validates_ranges() -> None:
    assert client.get("/api/scenario", params={"stores": 2}).status_code == 422
    assert client.get("/api/scenario", params={"stores": 13}).status_code == 422
    assert client.get("/api/scenario", params={"needs": 11}).status_code == 422
    assert client.get("/api/scenario", params={"products": 5}).status_code == 422


def test_baselines_shape() -> None:
    resp = client.get("/api/scenario/baselines", params={"seed": 9})
    assert resp.status_code == 200
    body = resp.json()
    expected_keys = {"routeKm", "itemCostCents", "storesUsed", "travelMinutes", "fitness"}
    for kind in ("random", "greedy"):
        assert kind in body
        assert expected_keys <= body[kind].keys()
        assert body[kind]["storesUsed"] >= 1
        assert body[kind]["routeKm"] > 0
    # Deterministic for a given seed.
    assert body == client.get("/api/scenario/baselines", params={"seed": 9}).json()


def test_problems_endpoint_lists_both() -> None:
    problems = client.get("/api/problems").json()["problems"]
    ids = {p["id"] for p in problems}
    assert {"allocation", "pickup"} <= ids
    pickup = next(p for p in problems if p["id"] == "pickup")
    assert {"seed", "stores", "products", "needs"} <= pickup["configSchema"].keys()


def test_pickup_run_streams_to_done() -> None:
    resp = client.post(
        "/api/runs",
        json={
            "problem": "pickup",
            "generations": 15,
            "populationSize": 30,
            "seed": 4,
            "problemConfig": {"seed": 4, "stores": 6, "products": 12, "needs": 5},
        },
    )
    assert resp.status_code == 201
    run_id = resp.json()["runId"]

    with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
        types: list[str] = []
        last = None
        while True:
            msg = ws.receive_json()
            types.append(msg["type"])
            last = msg
            if msg["type"] in ("done", "error"):
                break
    assert types[-1] == "done"
    assert "generation" in types
    spec = last["payload"]["renderSpec"]
    assert {"selection", "route", "routeKm", "itemCostCents",
            "storesUsed", "travelMinutes"} <= spec.keys()

"""Runner + API tests: streamed protocol shape and REST lifecycle."""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from app.main import app
from ga.engine import EngineConfig
from ga.problem import AllocationProblem
from ga.runner import RunSpec, stream_run


def _spec(generations: int = 30) -> RunSpec:
    problem = AllocationProblem(items=16, slots=16, seed=42)
    cfg = EngineConfig(population_size=40, generations=generations, seed=42)
    return RunSpec(problem=problem, config=cfg)


def test_stream_emits_generation_then_done() -> None:
    async def collect() -> list[dict]:
        return [m async for m in stream_run(_spec(20), max_events_per_sec=1000)]

    messages = asyncio.run(collect())
    types = [m["type"] for m in messages]
    assert types[-1] == "done"
    assert "generation" in types
    assert "error" not in types

    gen = next(m for m in messages if m["type"] == "generation")["payload"]
    assert {"gen", "bestFitness", "avgFitness", "worstFitness", "diversity",
            "bestGenome", "renderSpec"} <= gen.keys()

    done = messages[-1]["payload"]
    assert {"gen", "bestFitness", "bestGenome", "renderSpec", "elapsedMs"} <= done.keys()


def test_stream_throttles_event_count() -> None:
    async def collect() -> list[dict]:
        # 500 generations throttled to ~30/s should emit far fewer events.
        return [m async for m in stream_run(_spec(500), max_events_per_sec=30)]

    messages = asyncio.run(collect())
    gen_events = [m for m in messages if m["type"] == "generation"]
    assert len(gen_events) < 500
    assert messages[-1]["type"] == "done"


def test_stop_callback_ends_stream_early() -> None:
    state = {"count": 0}

    def should_stop() -> bool:
        state["count"] += 1
        return state["count"] > 5

    async def collect() -> list[dict]:
        return [
            m
            async for m in stream_run(
                _spec(500), should_stop=should_stop, max_events_per_sec=1000
            )
        ]

    messages = asyncio.run(collect())
    assert messages[-1]["type"] == "done"
    gen_events = [m for m in messages if m["type"] == "generation"]
    assert len(gen_events) < 500


client = TestClient(app)


def test_rest_run_lifecycle() -> None:
    resp = client.post(
        "/api/runs",
        json={"problem": "allocation", "generations": 15, "populationSize": 30, "seed": 1},
    )
    assert resp.status_code == 201
    run_id = resp.json()["runId"]

    got = client.get(f"/api/runs/{run_id}")
    assert got.status_code == 200
    assert got.json()["status"] == "running"

    stopped = client.post(f"/api/runs/{run_id}/stop")
    assert stopped.status_code == 200
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "stopped"


def test_create_run_validates_params() -> None:
    resp = client.post("/api/runs", json={"populationSize": 5})  # below min 20
    assert resp.status_code == 422


def test_get_unknown_run_404() -> None:
    assert client.get("/api/runs/does-not-exist").status_code == 404


def test_list_problems() -> None:
    resp = client.get("/api/problems")
    assert resp.status_code == 200
    problems = resp.json()["problems"]
    assert any(p["id"] == "allocation" for p in problems)


def test_websocket_streams_generation_and_done() -> None:
    resp = client.post(
        "/api/runs",
        json={"problem": "allocation", "generations": 12, "populationSize": 30, "seed": 3},
    )
    run_id = resp.json()["runId"]

    with client.websocket_connect(f"/api/runs/{run_id}/stream") as ws:
        types: list[str] = []
        while True:
            msg = ws.receive_json()
            types.append(msg["type"])
            if msg["type"] in ("done", "error"):
                break
    assert types[-1] == "done"
    assert "generation" in types
    assert client.get(f"/api/runs/{run_id}").json()["status"] == "done"

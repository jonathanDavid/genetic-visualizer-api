"""Run lifecycle: create, inspect, stop, and stream over WebSocket."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status

from ga.engine import EngineConfig
from ga.problem import build_problem
from ga.runner import RunSpec, stream_run

from ..registry import RunRecord, registry
from ..schemas import CreateRunResponse, RunParams, RunStatusResponse

router = APIRouter(prefix="/api", tags=["runs"])


def _engine_config(params: RunParams) -> EngineConfig:
    return EngineConfig(
        population_size=params.population_size,
        generations=params.generations,
        mutation_rate=params.mutation_rate,
        crossover_rate=params.crossover_rate,
        elitism=params.elitism,
        selection=params.selection,
        seed=params.seed,
    )


def _build_spec(params: RunParams) -> RunSpec:
    seed = params.seed if params.seed is not None else 12345
    problem = build_problem(params.problem, params.problem_config, seed)
    return RunSpec(problem=problem, config=_engine_config(params))


@router.post(
    "/runs",
    status_code=status.HTTP_201_CREATED,
    response_model=CreateRunResponse,
    summary="Create a run",
    description="Validates the problem config and registers the run. Nothing computes until the WebSocket stream connects.",
    responses={422: {"description": "Unknown problem or invalid problemConfig"}},
)
def create_run(params: RunParams) -> CreateRunResponse:
    try:
        build_problem(params.problem, params.problem_config, params.seed or 0)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record = registry.create(params)
    return CreateRunResponse(run_id=record.run_id)


@router.get(
    "/runs/{run_id}",
    response_model=RunStatusResponse,
    summary="Inspect a run",
    responses={404: {"description": "Run not found"}},
)
def get_run(run_id: str) -> RunStatusResponse:
    record = registry.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="run not found")
    return RunStatusResponse(
        run_id=record.run_id,
        status=record.status,
        params=record.params,
        best_fitness=record.best_fitness,
    )


@router.post(
    "/runs/{run_id}/stop",
    summary="Stop a running GA",
    description="Sets the stop flag; the stream finishes the current generation and closes with a final `done` message.",
    responses={404: {"description": "Run not found"}},
)
def stop_run(run_id: str) -> dict[str, str]:
    if not registry.stop(run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return {"status": "stopping"}


@router.websocket("/runs/{run_id}/stream")
async def stream(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    record = registry.get(run_id)
    if record is None:
        await websocket.send_json({"type": "error", "payload": {"message": "run not found"}})
        await websocket.close()
        return
    if record.started:
        await websocket.send_json(
            {"type": "error", "payload": {"message": "run already streaming"}}
        )
        await websocket.close()
        return

    record.started = True
    record.status = "running"
    spec = _build_spec(record.params)

    def on_progress(result) -> None:
        record.best_fitness = result.best_fitness

    try:
        async for message in stream_run(
            spec,
            should_stop=record.should_stop,
            on_progress=on_progress,
        ):
            await websocket.send_json(message)
        _finalize(record)
    except WebSocketDisconnect:
        record.stop_event.set()
        if record.status == "running":
            record.status = "stopped"
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass


def _finalize(record: RunRecord) -> None:
    if record.status != "stopped":
        record.status = "done"

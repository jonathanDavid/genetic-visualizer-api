"""Pydantic v2 models mirroring the shared architecture contract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RunParams(BaseModel):
    """Run configuration sent by the web client (contract §Run parameters)."""

    model_config = {"populate_by_name": True}

    problem: str = "allocation"
    population_size: int = Field(120, ge=20, le=500, alias="populationSize")
    generations: int = Field(200, ge=10, le=2000)
    mutation_rate: float = Field(0.05, ge=0.0, le=1.0, alias="mutationRate")
    crossover_rate: float = Field(0.9, ge=0.0, le=1.0, alias="crossoverRate")
    elitism: int = Field(2, ge=0, le=10)
    selection: Literal["tournament", "roulette"] = "tournament"
    seed: int | None = None
    problem_config: dict[str, Any] | None = Field(default=None, alias="problemConfig")


class CreateRunResponse(BaseModel):
    run_id: str = Field(alias="runId")

    model_config = {"populate_by_name": True}


class RunStatusResponse(BaseModel):
    run_id: str = Field(alias="runId")
    status: Literal["running", "done", "stopped"]
    params: RunParams
    best_fitness: float | None = Field(default=None, alias="bestFitness")

    model_config = {"populate_by_name": True}

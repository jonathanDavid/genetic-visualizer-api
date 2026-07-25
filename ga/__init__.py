"""Genetic-algorithm core: problems, engine and async runner."""

from .engine import EngineConfig, GenerationResult, GeneticEngine
from .pickup import PickupProblem, build_pickup_problem
from .problem import AllocationProblem, Problem, build_problem
from .runner import RunSpec, run_to_completion, stream_run
from .scenario import (
    Scenario,
    ScenarioValidationError,
    generate_scenario,
    scenario_from_dict,
)

__all__ = [
    "AllocationProblem",
    "PickupProblem",
    "build_pickup_problem",
    "Scenario",
    "ScenarioValidationError",
    "generate_scenario",
    "scenario_from_dict",
    "Problem",
    "build_problem",
    "EngineConfig",
    "GeneticEngine",
    "GenerationResult",
    "RunSpec",
    "stream_run",
    "run_to_completion",
]

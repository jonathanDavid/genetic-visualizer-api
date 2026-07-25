"""Genetic-algorithm core: problems, engine and async runner."""

from .engine import EngineConfig, GenerationResult, GeneticEngine
from .problem import AllocationProblem, Problem, build_problem
from .runner import RunSpec, run_to_completion, stream_run

__all__ = [
    "AllocationProblem",
    "Problem",
    "build_problem",
    "EngineConfig",
    "GeneticEngine",
    "GenerationResult",
    "RunSpec",
    "stream_run",
    "run_to_completion",
]

"""Async bridge between the CPU-bound engine and the WebSocket layer.

The GA loop is CPU-bound numpy work, so it runs in a worker thread; the async
side consumes generation events through an :class:`asyncio.Queue`. Events are
throttled to at most ``max_events_per_sec`` (default ~30/s) so a 2000-
generation run doesn't flood the socket — intermediate generations are
dropped, but the final generation is always delivered, followed by a ``done``
message.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import numpy as np

from .engine import EngineConfig, GeneticEngine, GenerationResult
from .problem import Problem


@dataclass(slots=True)
class RunSpec:
    problem: Problem
    config: EngineConfig


def _generation_message(problem: Problem, r: GenerationResult) -> dict[str, Any]:
    return {
        "type": "generation",
        "payload": {
            "gen": r.gen,
            "bestFitness": r.best_fitness,
            "avgFitness": r.avg_fitness,
            "worstFitness": r.worst_fitness,
            "diversity": r.diversity,
            "bestGenome": r.best_genome.tolist(),
            "renderSpec": problem.render_spec(r.best_genome),
        },
    }


def _done_message(
    problem: Problem, r: GenerationResult, elapsed_ms: float
) -> dict[str, Any]:
    return {
        "type": "done",
        "payload": {
            "gen": r.gen,
            "bestFitness": r.best_fitness,
            "bestGenome": r.best_genome.tolist(),
            "renderSpec": problem.render_spec(r.best_genome),
            "elapsedMs": round(elapsed_ms, 2),
        },
    }


async def stream_run(
    spec: RunSpec,
    *,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[GenerationResult], None] | None = None,
    max_events_per_sec: float = 30.0,
) -> AsyncIterator[dict[str, Any]]:
    """Run the engine in a thread and yield ``{type, payload}`` messages.

    Yields throttled ``generation`` events followed by exactly one ``done``
    (or an ``error``) message. ``on_progress`` is called for every generation
    (unthrottled) so the registry can track live best-fitness.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[GenerationResult | Exception | None] = asyncio.Queue(maxsize=256)
    engine = GeneticEngine(spec.problem, spec.config)

    def worker() -> None:
        try:
            for result in engine.run(should_stop=should_stop):
                loop.call_soon_threadsafe(queue.put_nowait, result)
        except Exception as exc:  # noqa: BLE001 - surfaced as an error event
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    thread = threading.Thread(target=worker, name="ga-worker", daemon=True)
    started = time.perf_counter()
    thread.start()

    min_interval = 1.0 / max_events_per_sec if max_events_per_sec > 0 else 0.0
    last_emit = 0.0
    last_result: GenerationResult | None = None

    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                yield {"type": "error", "payload": {"message": str(item)}}
                return
            last_result = item
            if on_progress is not None:
                on_progress(item)
            now = time.perf_counter()
            if now - last_emit >= min_interval:
                last_emit = now
                yield _generation_message(spec.problem, item)

        if last_result is not None:
            # Guarantee the terminal state is always sent, even if throttled.
            yield _generation_message(spec.problem, last_result)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            yield _done_message(spec.problem, last_result, elapsed_ms)
    finally:
        thread.join(timeout=1.0)


def run_to_completion(spec: RunSpec) -> GenerationResult:
    """Synchronous helper (used in tests): run fully, return the last result."""
    engine = GeneticEngine(spec.problem, spec.config)
    last: GenerationResult | None = None
    for last in engine.run():
        pass
    assert last is not None
    return last

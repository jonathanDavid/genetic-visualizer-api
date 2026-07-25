"""In-memory registry of runs with cooperative stop support.

A single process owns the registry. Each run has a threading.Event the GA
worker polls once per generation, so ``POST /api/runs/:id/stop`` halts a run
without killing the thread. State is intentionally ephemeral — see the README
"What I'd change at scale" section on persisting runs.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Literal

from .schemas import RunParams

RunStatus = Literal["running", "done", "stopped"]


@dataclass(slots=True)
class RunRecord:
    run_id: str
    params: RunParams
    status: RunStatus = "running"
    best_fitness: float | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    started: bool = False

    def should_stop(self) -> bool:
        return self.stop_event.is_set()


class RunRegistry:
    """Thread-safe store of runs keyed by id."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create(self, params: RunParams) -> RunRecord:
        run_id = uuid.uuid4().hex
        record = RunRecord(run_id=run_id, params=params)
        with self._lock:
            self._runs[run_id] = record
        return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def stop(self, run_id: str) -> bool:
        record = self.get(run_id)
        if record is None:
            return False
        record.stop_event.set()
        if record.status == "running":
            record.status = "stopped"
        return True


registry = RunRegistry()

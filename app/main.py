"""FastAPI application entrypoint.

Wires the run and problem routers, permissive CORS for the local web client,
and a health probe. Run locally with ``make dev`` / ``uvicorn app.main:app``.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import problems, runs, scenario

app = FastAPI(
    title="Genetic-Algorithm Visualizer API",
    version="1.0.0",
    summary="Streaming genetic-algorithm backend for store-item allocation.",
    description=(
        "A numpy GA that solves a grocery-**pickup** problem — one shopping list, several "
        "stores with their own stock and prices — and **streams every generation** to the "
        "visualizer.\n\n"
        "**The stream is a WebSocket** (OpenAPI cannot describe it, so it is documented "
        "here): connect to `ws://…/api/runs/{runId}/stream` after `POST /api/runs`; messages "
        "are `{type: 'progress'|'done'|'error', payload}` where progress carries the "
        "generation number, best fitness, and the best genome. One stream per run.\n\n"
        "Docs are regenerated from this code on every push — they cannot drift."
    ),
    openapi_tags=[
        {"name": "runs", "description": "Run lifecycle: create, inspect, stop (stream is the WebSocket above)"},
        {"name": "problems", "description": "Problem metadata the UI builds its controls from"},
        {"name": "scenario", "description": "The deterministic retail world + random/greedy baselines"},
        {"name": "meta", "description": "Health probe"},
    ],
)

_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:5200,http://localhost:5201",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(problems.router)
app.include_router(scenario.router)


@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}

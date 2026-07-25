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

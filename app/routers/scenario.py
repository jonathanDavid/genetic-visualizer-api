"""``GET /api/scenario`` and ``GET /api/scenario/baselines``.

The scenario endpoint returns the deterministic retail world the pickup GA
optimizes over, so the web client can draw the map before (and during) a run.
The baselines endpoint returns random and greedy reference plans in the same
metric shape the GA reports, powering the UI's comparison mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Query

from ga.pickup import PickupProblem
from ga.scenario import generate_scenario

router = APIRouter(prefix="/api", tags=["scenario"])


@router.get("/scenario")
def get_scenario(
    seed: int = Query(default=42, ge=0),
    stores: int = Query(default=6, ge=3, le=12),
    products: int = Query(default=12, ge=6, le=30),
    needs: int = Query(default=5, ge=3, le=10),
) -> dict[str, Any]:
    return generate_scenario(seed=seed, stores=stores, products=products, needs=needs).to_dict()


@router.get("/scenario/baselines")
def get_baselines(
    seed: int = Query(default=42, ge=0),
    stores: int = Query(default=6, ge=3, le=12),
    products: int = Query(default=12, ge=6, le=30),
    needs: int = Query(default=5, ge=3, le=10),
) -> dict[str, Any]:
    scenario = generate_scenario(seed=seed, stores=stores, products=products, needs=needs)
    problem = PickupProblem(scenario)

    # Deterministic random baseline: a seeded valid assignment.
    rng = np.random.default_rng(seed)
    random_genome = problem.random_genome(rng)
    greedy_genome = problem.greedy_genome()

    return {
        "random": problem.metrics(random_genome),
        "greedy": problem.metrics(greedy_genome),
    }

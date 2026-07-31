"""``GET /api/scenario`` and the baselines endpoints.

The scenario endpoint returns the deterministic retail world the pickup GA
optimizes over, so the web client can draw the map before (and during) a run.
The baselines endpoints return random and greedy reference plans in the same
metric shape the GA reports, powering the UI's comparison mode — for a seeded
demo scenario (``GET``) or for a caller's explicit real order (``POST``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Body, HTTPException, Query

from ga.pickup import PickupProblem
from ga.scenario import ScenarioValidationError, generate_scenario, scenario_from_dict

router = APIRouter(prefix="/api", tags=["scenario"])


def _baselines(problem: PickupProblem, seed: int) -> dict[str, Any]:
    # Deterministic random baseline: a seeded valid assignment.
    rng = np.random.default_rng(seed)
    return {
        "random": problem.metrics(problem.random_genome(rng)),
        "greedy": problem.metrics(problem.greedy_genome()),
    }


@router.get(
    "/scenario",
    summary="Generate the deterministic retail world",
    description="Seeded, so the same seed always yields the same stores/products/needs - the map the GA optimizes over.",
)
def get_scenario(
    seed: int = Query(default=42, ge=0),
    stores: int = Query(default=6, ge=3, le=12),
    products: int = Query(default=12, ge=6, le=30),
    needs: int = Query(default=5, ge=3, le=10),
) -> dict[str, Any]:
    return generate_scenario(seed=seed, stores=stores, products=products, needs=needs).to_dict()


@router.get(
    "/scenario/baselines",
    summary="Random + greedy reference plans (seeded scenario)",
    description="Same metric shape the GA reports, powering the UI comparison mode.",
)
def get_baselines(
    seed: int = Query(default=42, ge=0),
    stores: int = Query(default=6, ge=3, le=12),
    products: int = Query(default=12, ge=6, le=30),
    needs: int = Query(default=5, ge=3, le=10),
) -> dict[str, Any]:
    scenario = generate_scenario(seed=seed, stores=stores, products=products, needs=needs)
    return _baselines(PickupProblem(scenario), seed=seed)


@router.post(
    "/scenario/baselines",
    summary="Baselines for a caller-provided scenario",
    description="Send an explicit scenario document (e.g. a real order) and get the same random/greedy reference metrics.",
    responses={422: {"description": "Scenario document fails validation"}},
)
def post_baselines(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Baselines for a caller's explicit scenario (its own real order).

    Accepts either the scenario object directly or ``{ "scenario": {...} }``.
    Returns ``{ random, greedy }`` in the same metric shape as the GET form,
    and 422s with a clear message if the order is not coverable.
    """
    scenario_data = body.get("scenario", body) if isinstance(body, dict) else body
    try:
        scenario = scenario_from_dict(scenario_data)
    except ScenarioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _baselines(PickupProblem(scenario), seed=0)

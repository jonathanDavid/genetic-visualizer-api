"""``GET /api/problems`` — metadata so the UI can build its controls."""

from __future__ import annotations

from fastapi import APIRouter

from ga.problem import PROBLEMS

router = APIRouter(prefix="/api", tags=["problems"])


@router.get(
    "/problems",
    summary="List available problems",
    description="Each problem describes its tunable parameters so the UI renders matching controls.",
)
def list_problems() -> dict[str, list[dict]]:
    return {"problems": [p.describe() for p in PROBLEMS.values()]}

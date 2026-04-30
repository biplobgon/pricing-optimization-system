"""Demand forecast API route placeholders."""

from fastapi import APIRouter


router = APIRouter(prefix="/api/v1/forecasts", tags=["forecasts"])


@router.post("/demand")
def forecast_demand() -> dict[str, str]:
    """Return demand forecasts."""
    return {"status": "not_implemented"}

"""Product API route placeholders."""

from fastapi import APIRouter


router = APIRouter(prefix="/api/v1/products", tags=["products"])


@router.get("")
def list_products() -> list[dict]:
    """Return product records."""
    return []

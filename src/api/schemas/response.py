"""Response schema placeholders."""

from typing import Optional

from pydantic import BaseModel


class PriceRecommendationResponse(BaseModel):
    """Output payload for price recommendations."""

    product_id: str
    recommended_price: float
    expected_revenue: Optional[float] = None

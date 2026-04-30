"""Request schema placeholders."""

from typing import Optional

from pydantic import BaseModel


class PriceRecommendationRequest(BaseModel):
    """Input payload for price recommendations."""

    product_id: str
    current_price: float
    inventory_level: Optional[int] = None

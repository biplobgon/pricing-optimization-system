"""Dynamic Pricer Module.

Implements dynamic pricing that adjusts based on real-time factors.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta


class DynamicPricer:
    """Dynamic pricing system that adjusts prices based on market conditions.
    
    Attributes:
        base_pricer: Base optimizer
        update_frequency: How often to update prices
    """
    
    def __init__(self, update_frequency: str = "daily"):
        """Initialize the DynamicPricer.
        
        Args:
            update_frequency: Frequency of price updates
        """
        self.update_frequency = update_frequency
        self.price_history = {}
        self.current_prices = {}
        
    def calculate_dynamic_price(self, product_id: str, base_price: float,
                                demand_forecast: float, inventory_level: float,
                                competitor_price: float, elasticity: float,
                                cost: float) -> Dict:
        """Calculate dynamic price for a product.
        
        Args:
            product_id: Product identifier
            base_price: Base price
            demand_forecast: Forecasted demand
            inventory_level: Current inventory
            competitor_price: Competitor price
            elasticity: Price elasticity
            cost: Unit cost
            
        Returns:
            Dictionary with price recommendation
        """
        # Inventory adjustment factor
        if inventory_level > demand_forecast * 2:
            inventory_factor = 0.9  # High inventory - reduce price
        elif inventory_level < demand_forecast * 0.5:
            inventory_factor = 1.1  # Low inventory - increase price
        else:
            inventory_factor = 1.0  # Normal inventory
            
        # Competition adjustment factor
        if competitor_price:
            comp_ratio = base_price / competitor_price
            if comp_ratio > 1.1:
                comp_factor = 0.95  # Price too high vs competition
            elif comp_ratio < 0.9:
                comp_factor = 1.05  # Price too low vs competition
            else:
                comp_factor = 1.0
        else:
            comp_factor = 1.0
            
        # Calculate adjusted price
        adjusted_price = base_price * inventory_factor * comp_factor
        
        # Ensure price is above cost
        adjusted_price = max(adjusted_price, cost * 1.1)
        
        # Calculate expected metrics
        expected_demand = demand_forecast * (adjusted_price / base_price) ** elasticity
        expected_revenue = adjusted_price * expected_demand
        expected_profit = (adjusted_price - cost) * expected_demand
        
        result = {
            "product_id": product_id,
            "recommended_price": round(adjusted_price, 2),
            "base_price": base_price,
            "adjustment_factors": {
                "inventory_factor": inventory_factor,
                "competition_factor": comp_factor
            },
            "expected_demand": round(expected_demand, 0),
            "expected_revenue": round(expected_revenue, 2),
            "expected_profit": round(expected_profit, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store in history
        if product_id not in self.price_history:
            self.price_history[product_id] = []
        self.price_history[product_id].append(result)
        
        return result
    
    def get_price_trajectory(self, product_id: str) -> pd.DataFrame:
        """Get price history for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            DataFrame with price history
        """
        if product_id not in self.price_history:
            return pd.DataFrame()
            
        return pd.DataFrame(self.price_history[product_id])
    
    def get_current_price(self, product_id: str) -> Optional[float]:
        """Get current price for a product.
        
        Args:
            product_id: Product identifier
            
        Returns:
            Current price
        """
        if product_id in self.price_history and self.price_history[product_id]:
            return self.price_history[product_id][-1]["recommended_price"]
        return None
    
    def bulk_calculate(self, products: List[Dict]) -> List[Dict]:
        """Calculate dynamic prices for multiple products.
        
        Args:
            products: List of product dictionaries
            
        Returns:
            List of price recommendations
        """
        results = []
        
        for product in products:
            result = self.calculate_dynamic_price(
                product_id=product["product_id"],
                base_price=product["base_price"],
                demand_forecast=product["demand_forecast"],
                inventory_level=product["inventory_level"],
                competitor_price=product.get("competitor_price"),
                elasticity=product.get("elasticity", -1.0),
                cost=product["cost"]
            )
            results.append(result)
            
        return results
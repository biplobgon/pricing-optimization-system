"""Optimizer Module.

Optimizes prices to maximize revenue based on demand forecasts and elasticity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, Callable


class Optimizer:
    """Optimize prices to maximize revenue/profit.
    
    Attributes:
        objective: Optimization objective (revenue, profit, volume)
    """
    
    def __init__(self, objective: str = "revenue"):
        """Initialize the Optimizer.
        
        Args:
            objective: Optimization objective
        """
        self.objective = objective
        self.optimal_price = None
        self.optimization_result = None
        
    def optimize(self, demand_func: Callable, price_range: Tuple[float, float],
                 cost: float = 0, constraints: Optional[Dict] = None) -> Dict:
        """Optimize price to maximize objective.
        
        Args:
            demand_func: Function that predicts demand given price
            price_range: (min_price, max_price)
            cost: Unit cost
            constraints: Additional constraints
            
        Returns:
            Dictionary with optimization results
        """
        min_price, max_price = price_range
        best_price = min_price
        best_value = -np.inf
        
        # Grid search for optimal price
        price_steps = 100
        step_size = (max_price - min_price) / price_steps
        
        for price in np.arange(min_price, max_price, step_size):
            demand = demand_func(price)
            
            if self.objective == "revenue":
                value = price * demand
            elif self.objective == "profit":
                value = (price - cost) * demand
            elif self.objective == "volume":
                value = demand
                
            if value > best_value:
                best_value = value
                best_price = price
                
        self.optimal_price = best_price
        self.optimization_result = {
            "optimal_price": best_price,
            "expected_demand": demand_func(best_price),
            "expected_value": best_value,
            "objective": self.objective
        }
        
        return self.optimization_result
    
    def optimize_with_elasticity(self, base_price: float, base_demand: float,
                                 elasticity: float, price_range: Tuple[float, float],
                                 cost: float = 0) -> Dict:
        """Optimize price using elasticity.
        
        Args:
            base_price: Current price
            base_demand: Current demand
            elasticity: Price elasticity
            price_range: (min_price, max_price)
            cost: Unit cost
            
        Returns:
            Dictionary with optimization results
        """
        # Demand function based on elasticity
        def demand_func(price):
            price_ratio = price / base_price
            return base_demand * (price_ratio ** elasticity)
            
        return self.optimize(demand_func, price_range, cost)
    
    def optimize_with_constraints(self, demand_func: Callable, 
                                  price_range: Tuple[float, float],
                                  cost: float, min_margin: float = 0.1,
                                  max_price_change: float = 0.2,
                                  current_price: float = None) -> Dict:
        """Optimize price with business constraints.
        
        Args:
            demand_func: Function that predicts demand given price
            price_range: (min_price, max_price)
            cost: Unit cost
            min_margin: Minimum margin requirement
            max_price_change: Maximum price change from current
            current_price: Current price
            
        Returns:
            Dictionary with optimization results
        """
        min_price, max_price = price_range
        
        # Apply constraints
        if current_price is not None:
            min_price = max(min_price, current_price * (1 - max_price_change))
            max_price = min(max_price, current_price * (1 + max_price_change))
            
        # Minimum margin constraint
        min_price = max(min_price, cost * (1 + min_margin))
        
        return self.optimize(demand_func, (min_price, max_price), cost)
    
    def get_optimal_price(self) -> float:
        """Get the optimal price.
        
        Returns:
            Optimal price
        """
        return self.optimal_price
    
    def get_optimization_summary(self) -> Dict:
        """Get optimization summary.
        
        Returns:
            Dictionary with summary
        """
        return self.optimization_result or {}
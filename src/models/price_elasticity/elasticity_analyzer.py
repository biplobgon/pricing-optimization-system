"""Elasticity Analyzer Module.

Analyzes price elasticity results and provides insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class ElasticityAnalyzer:
    """Analyze price elasticity results.
    
    Attributes:
        elasticity_estimator: ElasticityEstimator instance
    """
    
    def __init__(self):
        """Initialize the ElasticityAnalyzer."""
        self.elasticity_results = {}
        
    def analyze_by_category(self, df: pd.DataFrame, category_col: str = "category",
                           price_col: str = "avg_price", demand_col: str = "demand") -> Dict:
        """Analyze elasticity by product category.
        
        Args:
            df: DataFrame with data
            category_col: Category column
            price_col: Price column
            demand_col: Demand column
            
        Returns:
            Dictionary with elasticity by category
        """
        from .elasticity_estimator import ElasticityEstimator
        
        results = {}
        
        for category in df[category_col].unique():
            subset = df[df[category_col] == category]
            estimator = ElasticityEstimator()
            estimator.fit(subset, price_col, demand_col)
            results[category] = {
                "elasticity": estimator.get_elasticity(),
                "category": estimator.get_elasticity_category(),
                "count": len(subset)
            }
            
        self.elasticity_results = results
        return results
    
    def analyze_by_product(self, df: pd.DataFrame, product_col: str = "product_id",
                           price_col: str = "avg_price", demand_col: str = "demand") -> Dict:
        """Analyze elasticity by product.
        
        Args:
            df: DataFrame with data
            product_col: Product column
            price_col: Price column
            demand_col: Demand column
            
        Returns:
            Dictionary with elasticity by product
        """
        from .elasticity_estimator import ElasticityEstimator
        
        results = {}
        
        for product in df[product_col].unique():
            subset = df[df[product_col] == product]
            if len(subset) >= 10:  # Minimum data points
                estimator = ElasticityEstimator()
                estimator.fit(subset, price_col, demand_col)
                results[product] = {
                    "elasticity": estimator.get_elasticity(),
                    "category": estimator.get_elasticity_category()
                }
                
        self.elasticity_results = results
        return results
    
    def get_elasticity_distribution(self) -> Dict:
        """Get distribution of elasticity values.
        
        Returns:
            Dictionary with distribution statistics
        """
        if not self.elasticity_results:
            return {}
            
        elasticities = [v["elasticity"] for v in self.elasticity_results.values() if v.get("elasticity")]
        
        return {
            "mean": np.mean(elasticities),
            "median": np.median(elasticities),
            "std": np.std(elasticities),
            "min": np.min(elasticities),
            "max": np.max(elasticities),
            "count": len(elasticities)
        }
    
    def identify_elastic_products(self, threshold: float = -1.0) -> List[str]:
        """Identify products with high elasticity.
        
        Args:
            threshold: Elasticity threshold
            
        Returns:
            List of product IDs
        """
        elastic_products = []
        
        for product, data in self.elasticity_results.items():
            if data.get("elasticity", 0) < threshold:
                elastic_products.append(product)
                
        return elastic_products
    
    def identify_inelastic_products(self, threshold: float = -0.5) -> List[str]:
        """Identify products with low elasticity.
        
        Args:
            threshold: Elasticity threshold
            
        Returns:
            List of product IDs
        """
        inelastic_products = []
        
        for product, data in self.elasticity_results.items():
            if data.get("elasticity", 0) > threshold:
                inelastic_products.append(product)
                
        return inelastic_products
    
    def get_pricing_recommendations(self) -> Dict:
        """Get pricing recommendations based on elasticity.
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {}
        
        for product, data in self.elasticity_results.items():
            elasticity = data.get("elasticity", 0)
            
            if elasticity < -1:
                recommendations[product] = {
                    "strategy": "Decrease price to increase revenue",
                    "action": "Price reduction",
                    "rationale": "Elastic demand - price cut will increase volume"
                }
            elif elasticity > -1:
                recommendations[product] = {
                    "strategy": "Increase price to maximize revenue",
                    "action": "Price increase",
                    "rationale": "Inelastic demand - price increase won't hurt volume"
                }
            else:
                recommendations[product] = {
                    "strategy": "Maintain current price",
                    "action": "Hold",
                    "rationale": "Unit elastic - price change won't affect revenue"
                }
                
        return recommendations
"""Segment Elasticity Module.

Estimates price elasticity for different customer segments.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .elasticity_estimator import ElasticityEstimator


class SegmentElasticity:
    """Estimate price elasticity for customer segments.
    
    Attributes:
        segment_column: Column to segment by
    """
    
    def __init__(self, segment_column: str = "region"):
        """Initialize the SegmentElasticity.
        
        Args:
            segment_column: Column to segment by
        """
        self.segment_column = segment_column
        self.segment_elasticities = {}
        
    def fit(self, df: pd.DataFrame, price_col: str = "avg_price", 
            demand_col: str = "demand") -> 'SegmentElasticity':
        """Fit elasticity models for each segment.
        
        Args:
            df: DataFrame with data
            price_col: Price column
            demand_col: Demand column
            
        Returns:
            Self
        """
        for segment in df[self.segment_column].unique():
            subset = df[df[self.segment_column] == segment]
            
            if len(subset) >= 10:
                estimator = ElasticityEstimator()
                estimator.fit(subset, price_col, demand_col)
                
                self.segment_elasticities[segment] = {
                    "elasticity": estimator.get_elasticity(),
                    "category": estimator.get_elasticity_category(),
                    "count": len(subset)
                }
                
        return self
    
    def get_segment_elasticity(self, segment: str) -> float:
        """Get elasticity for a specific segment.
        
        Args:
            segment: Segment name
            
        Returns:
            Elasticity value
        """
        return self.segment_elasticities.get(segment, {}).get("elasticity", 0)
    
    def compare_segments(self) -> pd.DataFrame:
        """Compare elasticity across segments.
        
        Returns:
            DataFrame with segment comparison
        """
        data = []
        for segment, info in self.segment_elasticities.items():
            data.append({
                "segment": segment,
                "elasticity": info["elasticity"],
                "category": info["category"],
                "count": info["count"]
            })
            
        return pd.DataFrame(data).sort_values("elasticity")
    
    def identify_most_elastic_segment(self) -> str:
        """Identify the most price-sensitive segment.
        
        Returns:
            Segment name
        """
        if not self.segment_elasticities:
            return ""
            
        min_elasticity = min(self.segment_elasticities.items(), key=lambda x: x[1]["elasticity"])
        return min_elasticity[0]
    
    def identify_least_elastic_segment(self) -> str:
        """Identify the least price-sensitive segment.
        
        Returns:
            Segment name
        """
        if not self.segment_elasticities:
            return ""
            
        max_elasticity = max(self.segment_elasticities.items(), key=lambda x: x[1]["elasticity"])
        return max_elasticity[0]
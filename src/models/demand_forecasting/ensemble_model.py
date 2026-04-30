"""Ensemble Model for Demand Forecasting.

Combines multiple forecasting models for improved accuracy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .arima_model import ARIMAModel
from .prophet_model import ProphetModel
from .xgboost_model import XGBoostModel


class EnsembleModel:
    """Ensemble model combining multiple forecasting approaches.
    
    Attributes:
        models: List of models to ensemble
        weights: Weights for each model
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        """Initialize the EnsembleModel.
        
        Args:
            weights: Weights for each model
        """
        self.weights = weights
        self.models = []
        
    def add_model(self, model, weight: float = 1.0):
        """Add a model to the ensemble.
        
        Args:
            model: Model instance
            weight: Weight for this model
        """
        self.models.append((model, weight))
        
    def fit(self, df: pd.DataFrame, target_col: str = "demand", 
            time_col: str = "month_year") -> 'EnsembleModel':
        """Fit all models in the ensemble.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            time_col: Time column name
            
        Returns:
            Self
        """
        for model, _ in self.models:
            if hasattr(model, 'fit'):
                model.fit(df, target_col=target_col, time_col=time_col)
                
        return self
    
    def predict(self, df: pd.DataFrame, target_col: str = "demand",
                time_col: str = "month_year", steps: int = 1) -> np.ndarray:
        """Generate ensemble predictions.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            time_col: Time column name
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        predictions = []
        
        for model, _ in self.models:
            if hasattr(model, 'forecast'):
                pred = model.forecast(steps)
                predictions.append(pred)
            elif hasattr(model, 'predict'):
                pred = model.predict(df)
                predictions.append(pred)
                
        if not predictions:
            raise ValueError("No models in ensemble")
            
        # Weighted average
        if self.weights:
            weights = self.weights[:len(predictions)]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            result = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                result += weights[i] * pred
            return result
        else:
            return np.mean(predictions, axis=0)
    
    def get_model_performance(self) -> Dict:
        """Get performance metrics for each model.
        
        Returns:
            Dictionary with model performance
        """
        performance = {}
        for i, (model, weight) in enumerate(self.models):
            model_name = model.__class__.__name__
            performance[f"model_{i}_{model_name}"] = {"weight": weight}
            
        return performance
"""ARIMA Model for Demand Forecasting.

Implements ARIMA/SARIMA models for time series demand forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class ARIMAModel:
    """ARIMA/SARIMA model for demand forecasting.
    
    Attributes:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s)
    """
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0), 
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None):
        """Initialize the ARIMAModel.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def fit(self, df: pd.DataFrame, target_col: str = "demand", 
            time_col: str = "month_year") -> 'ARIMAModel':
        """Fit the ARIMA model.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            time_col: Time column name
            
        Returns:
            Self
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA model")
            
        # Prepare data
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        series = df[target_col].values
        
        # Fit model
        if self.seasonal_order:
            self.model = SARIMAX(
                series, 
                order=self.order, 
                seasonal_order=self.seasonal_order
            )
        else:
            self.model = ARIMA(series, order=self.order)
            
        self.fitted_model = self.model.fit(disp=False)
        
        return self
    
    def forecast(self, steps: int) -> np.ndarray:
        """Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def get_model_summary(self) -> Dict:
        """Get model summary statistics.
        
        Returns:
            Dictionary containing model summary
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return {
            "aic": self.fitted_model.aic,
            "bic": self.fitted_model.bic,
            "log_likelihood": self.fitted_model.llf,
            "order": self.order,
            "seasonal_order": self.seasonal_order
        }
    
    def predict_in_sample(self) -> np.ndarray:
        """Get in-sample predictions.
        
        Returns:
            Array of in-sample predictions
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.fitted_model.fittedvalues
    
    def get_residuals(self) -> np.ndarray:
        """Get model residuals.
        
        Returns:
            Array of residuals
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.fitted_model.resid
"""Prophet Model for Demand Forecasting.

Implements Facebook Prophet for time series demand forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetModel:
    """Prophet model for demand forecasting.
    
    Attributes:
        yearly_seasonality: Include yearly seasonality
        weekly_seasonality: Include weekly seasonality
        daily_seasonality: Include daily seasonality
    """
    
    def __init__(self, yearly_seasonality: bool = True, 
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 changepoint_prior_scale: float = 0.05):
        """Initialize the ProphetModel.
        
        Args:
            yearly_seasonality: Include yearly seasonality
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            changepoint_prior_scale: Changepoint prior scale
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model = None
        
    def fit(self, df: pd.DataFrame, target_col: str = "demand", 
            time_col: str = "month_year") -> 'ProphetModel':
        """Fit the Prophet model.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            time_col: Time column name
            
        Returns:
            Self
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet is required for Prophet model")
            
        # Prepare data for Prophet
        prophet_df = df.copy()
        prophet_df[time_col] = pd.to_datetime(prophet_df[time_col])
        prophet_df = prophet_df.sort_values(time_col)
        
        # Prophet requires columns named 'ds' and 'y'
        prophet_df = prophet_df.rename(columns={
            time_col: 'ds',
            target_col: 'y'
        })
        
        # Initialize and fit model
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        self.model.fit(prophet_df)
        
        return self
    
    def forecast(self, periods: int, freq: str = "MS") -> pd.DataFrame:
        """Generate forecasts.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of the data
            
        Returns:
            DataFrame with forecasts
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        return forecast
    
    def get_components(self) -> pd.DataFrame:
        """Get trend, seasonality components.
        
        Returns:
            DataFrame with components
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.model.plot_components(self.model.predict(
            self.model.make_future_dataframe(1)
        ))
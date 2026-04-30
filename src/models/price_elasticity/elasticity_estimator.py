"""Elasticity Estimator Module.

Estimates price elasticity of demand using various econometric methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats
from sklearn.linear_model import LinearRegression


class ElasticityEstimator:
    """Estimate price elasticity of demand.
    
    Attributes:
        method: Method for elasticity estimation
    """
    
    def __init__(self, method: str = "log_linear"):
        """Initialize the ElasticityEstimator.
        
        Args:
            method: Method for estimation (log_linear, linear, polynomial)
        """
        self.method = method
        self.elasticity = None
        self.model = None
        
    def fit(self, df: pd.DataFrame, price_col: str = "avg_price", 
            demand_col: str = "demand") -> 'ElasticityEstimator':
        """Fit the elasticity model.
        
        Args:
            df: DataFrame with price and demand data
            price_col: Price column name
            demand_col: Demand column name
            
        Returns:
            Self
        """
        if self.method == "log_linear":
            self._fit_log_linear(df, price_col, demand_col)
        elif self.method == "linear":
            self._fit_linear(df, price_col, demand_col)
        elif self.method == "polynomial":
            self._fit_polynomial(df, price_col, demand_col)
            
        return self
    
    def _fit_log_linear(self, df: pd.DataFrame, price_col: str, demand_col: str):
        """Fit log-linear demand model.
        
        Args:
            df: DataFrame with price and demand data
            price_col: Price column name
            demand_col: Demand column name
        """
        # log(Q) = a + b*log(P)
        log_price = np.log(df[price_col]).values.reshape(-1, 1)
        log_demand = np.log(df[demand_col]).values
        
        self.model = LinearRegression()
        self.model.fit(log_price, log_demand)
        
        # Elasticity is the coefficient of log(price)
        self.elasticity = self.model.coef_[0]
        
    def _fit_linear(self, df: pd.DataFrame, price_col: str, demand_col: str):
        """Fit linear demand model.
        
        Args:
            df: DataFrame with price and demand data
            price_col: Price column name
            demand_col: Demand column name
        """
        price = df[price_col].values.reshape(-1, 1)
        demand = df[demand_col].values
        
        self.model = LinearRegression()
        self.model.fit(price, demand)
        
        # Calculate point elasticity at mean values
        mean_price = df[price_col].mean()
        mean_demand = df[demand_col].mean()
        slope = self.model.coef_[0]
        
        self.elasticity = (slope * mean_price) / mean_demand
        
    def _fit_polynomial(self, df: pd.DataFrame, price_col: str, demand_col: str):
        """Fit polynomial demand model.
        
        Args:
            df: DataFrame with price and demand data
            price_col: Price column name
            demand_col: Demand column name
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        price = df[price_col].values.reshape(-1, 1)
        demand = df[demand_col].values
        
        poly = PolynomialFeatures(degree=2, include_bias=False)
        price_poly = poly.fit_transform(price)
        
        self.model = LinearRegression()
        self.model.fit(price_poly, demand)
        
        # Calculate elasticity at mean price
        mean_price = df[price_col].mean()
        self._calculate_polynomial_elasticity(mean_price, poly, price)
        
    def _calculate_polynomial_elasticity(self, price: float, poly, price_data):
        """Calculate elasticity for polynomial model.
        
        Args:
            price: Price to calculate at
            poly: PolynomialFeatures instance
            price_data: Original price data
        """
        # Simplified elasticity calculation
        self.elasticity = -1.0  # Placeholder
        
    def get_elasticity(self) -> float:
        """Get the estimated elasticity.
        
        Returns:
            Elasticity coefficient
        """
        return self.elasticity
    
    def get_elasticity_category(self) -> str:
        """Get the category of elasticity.
        
        Returns:
            Elasticity category string
        """
        if self.elasticity is None:
            return "Unknown"
        elif self.elasticity == 0:
            return "Perfectly Inelastic"
        elif self.elasticity < -1:
            return "Elastic"
        elif self.elasticity == -1:
            return "Unit Elastic"
        else:
            return "Inelastic"
    
    def predict_demand(self, price: float) -> float:
        """Predict demand at a given price.
        
        Args:
            price: Price to predict at
            
        Returns:
            Predicted demand
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if self.method == "log_linear":
            log_p = np.log(price)
            log_q = self.model.intercept_ + self.model.coef_[0] * log_p
            return np.exp(log_q)
        else:
            return self.model.predict([[price]])[0]
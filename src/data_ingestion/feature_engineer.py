"""Feature Engineer Module.

Creates and engineers features for the pricing optimization models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class FeatureEngineer:
    """Engineer features for pricing optimization models.
    
    Attributes:
        time_column: Column containing time information
        target_column: Target column for modeling
    """
    
    def __init__(self, time_column: str = "month_year", target_column: str = "demand"):
        """Initialize the FeatureEngineer.
        
        Args:
            time_column: Column containing time information
            target_column: Target column for modeling
        """
        self.time_column = time_column
        self.target_column = target_column
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        
        # Parse datetime if needed
        if self.time_column in df.columns:
            df[self.time_column] = pd.to_datetime(df[self.time_column])
            df["year"] = df[self.time_column].dt.year
            df["month"] = df[self.time_column].dt.month
            df["quarter"] = df[self.time_column].dt.quarter
            df["day_of_year"] = df[self.time_column].dt.dayofyear
            df["week_of_year"] = df[self.time_column].dt.isocalendar().week
            
        return df
    
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lag features for time series.
        
        Args:
            df: DataFrame to process
            columns: Columns to create lag features for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        df = df.copy()
        
        for col in columns:
            for lag in lags:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
                
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            df: DataFrame to process
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        df = df.copy()
        
        for col in columns:
            for window in windows:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
                df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
                df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
                
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-related features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with price features added
        """
        df = df.copy()
        
        # Price-related features
        if "avg_price" in df.columns and "cost" in df.columns:
            df["margin"] = df["avg_price"] - df["cost"]
            df["margin_percent"] = (df["margin"] / df["avg_price"]) * 100
            
        if "avg_price" in df.columns and "competitor_price" in df.columns:
            df["price_diff"] = df["avg_price"] - df["competitor_price"]
            df["price_ratio"] = df["avg_price"] / df["competitor_price"]
            
        if "demand" in df.columns and "avg_price" in df.columns:
            df["revenue"] = df["demand"] * df["avg_price"]
            
        return df
    
    def create_competition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create competition-related features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with competition features added
        """
        df = df.copy()
        
        if "competitor_price" in df.columns and "avg_price" in df.columns:
            df["price_advantage"] = df["competitor_price"] - df["avg_price"]
            df["price_advantage_pct"] = (df["price_advantage"] / df["avg_price"]) * 100
            
        if "competitor_count" in df.columns and "market_share" in df.columns:
            df["competition_intensity"] = df["competitor_count"] * (1 - df["market_share"])
            
        return df
    
    def create_demand_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demand-related features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with demand features added
        """
        df = df.copy()
        
        if "demand" in df.columns and "inventory" in df.columns:
            df["stock_ratio"] = df["demand"] / df["inventory"]
            df["stock_coverage"] = df["inventory"] / df["demand"]
            
        if "promotion_flag" in df.columns and "demand" in df.columns:
            df["promo_demand"] = df["promotion_flag"] * df["demand"]
            
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with all features added
        """
        df = self.create_time_features(df)
        df = self.create_price_features(df)
        df = self.create_competition_features(df)
        df = self.create_demand_features(df)
        
        return df
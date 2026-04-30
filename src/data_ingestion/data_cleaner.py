"""Data Cleaner Module.

Handles data cleaning operations including missing value treatment,
outlier detection, and data normalization.
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict


class DataCleaner:
    """Clean and preprocess data for the pricing optimization system.
    
    Attributes:
        missing_strategy: Strategy for handling missing values
        outlier_method: Method for detecting outliers
    """
    
    def __init__(self, missing_strategy: str = "mean", outlier_method: str = "iqr"):
        """Initialize the DataCleaner.
        
        Args:
            missing_strategy: Strategy for handling missing values
            outlier_method: Method for detecting outliers
        """
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        
    def handle_missing_values(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: DataFrame to clean
            columns: Columns to process (None = all columns)
            
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        columns = columns or df.columns
        
        for col in columns:
            if df[col].isnull().any():
                if self.missing_strategy == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif self.missing_strategy == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif self.missing_strategy == "mode":
                    df[col].fillna(df[col].mode()[0], inplace=True)
                elif self.missing_strategy == "forward_fill":
                    df[col].fillna(method="ffill", inplace=True)
                elif self.missing_strategy == "backward_fill":
                    df[col].fillna(method="bfill", inplace=True)
                    
        return df
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str], threshold: float = 1.5) -> pd.DataFrame:
        """Remove outliers from the dataset.
        
        Args:
            df: DataFrame to clean
            columns: Columns to check for outliers
            threshold: IQR multiplier for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        df = df.copy()
        
        for col in columns:
            if df[col].dtype in [np.float64, np.int64]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
        return df
    
    def normalize_column(self, df: pd.DataFrame, column: str, method: str = "min_max") -> pd.DataFrame:
        """Normalize a numeric column.
        
        Args:
            df: DataFrame to process
            column: Column to normalize
            method: Normalization method (min_max or z_score)
            
        Returns:
            DataFrame with normalized column
        """
        df = df.copy()
        
        if method == "min_max":
            min_val = df[column].min()
            max_val = df[column].max()
            df[column] = (df[column] - min_val) / (max_val - min_val)
        elif method == "z_score":
            mean = df[column].mean()
            std = df[column].std()
            df[column] = (df[column] - mean) / std
            
        return df
    
    def clean_categorical(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Clean categorical data.
        
        Args:
            df: DataFrame to clean
            column: Categorical column to clean
            
        Returns:
            DataFrame with cleaned categorical data
        """
        df = df.copy()
        df[column] = df[column].str.strip().str.lower()
        return df
    
    def get_cleaning_report(self, df: pd.DataFrame) -> Dict:
        """Generate a cleaning report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing cleaning statistics
        """
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.to_dict()
        }
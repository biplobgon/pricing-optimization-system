"""XGBoost Model for Demand Forecasting.

Implements XGBoost for demand forecasting with engineered features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostModel:
    """XGBoost model for demand forecasting.
    
    Attributes:
        params: XGBoost parameters
        n_estimators: Number of boosting rounds
    """
    
    def __init__(self, n_estimators: int = 100, 
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 **kwargs):
        """Initialize the XGBoostModel.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            **kwargs: Additional XGBoost parameters
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.params = kwargs
        self.model = None
        
    def fit(self, df: pd.DataFrame, target_col: str = "demand", 
            feature_cols: Optional[List[str]] = None) -> 'XGBoostModel':
        """Fit the XGBoost model.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            feature_cols: Feature column names
            
        Returns:
            Self
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is required for XGBoost model")
            
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]
            
        X = df[feature_cols]
        y = df[target_col]
        
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.params
        )
        
        self.model.fit(X, y)
        
        return self
    
    def predict(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Generate predictions.
        
        Args:
            df: DataFrame with features
            feature_cols: Feature column names
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if feature_cols is None:
            feature_cols = [c for c in df.columns]
            
        return self.model.predict(df[feature_cols])
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance scores.
        
        Returns:
            Dictionary of feature importance
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        importance = self.model.feature_importances_
        features = self.model.feature_names_in_
        
        return dict(zip(features, importance))
    
    def cross_validate(self, df: pd.DataFrame, target_col: str = "demand",
                       feature_cols: Optional[List[str]] = None,
                       cv: int = 5) -> Dict:
        """Perform cross-validation.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            feature_cols: Feature column names
            cv: Number of folds
            
        Returns:
            Dictionary with CV results
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]
            
        X = df[feature_cols]
        y = df[target_col]
        
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(
            self.model, X, y, 
            cv=cv, scoring='neg_mean_squared_error'
        )
        
        return {
            "cv_scores": -scores,
            "mean_score": -scores.mean(),
            "std_score": scores.std()
        }
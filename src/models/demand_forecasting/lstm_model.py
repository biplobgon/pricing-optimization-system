"""LSTM Model for Demand Forecasting.

Implements LSTM neural network for time series demand forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


class LSTMModel:
    """LSTM model for demand forecasting.
    
    Attributes:
        units: Number of LSTM units
        layers: Number of LSTM layers
        dropout: Dropout rate
    """
    
    def __init__(self, units: int = 50, layers: int = 1, dropout: float = 0.2,
                 epochs: int = 50, batch_size: int = 32):
        """Initialize the LSTMModel.
        
        Args:
            units: Number of LSTM units
            layers: Number of LSTM layers
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size
        """
        self.units = units
        self.layers = layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM.
        
        Args:
            data: Input data
            seq_length: Sequence length
            
        Returns:
            X and y arrays
        """
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def fit(self, df: pd.DataFrame, target_col: str = "demand", 
            time_col: str = "month_year", seq_length: int = 10) -> 'LSTMModel':
        """Fit the LSTM model.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            time_col: Time column name
            seq_length: Sequence length for LSTM
            
        Returns:
            Self
        """
        if not KERAS_AVAILABLE:
            raise ImportError("tensorflow is required for LSTM model")
            
        # Prepare data
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        
        # Scale data
        from sklearn.preprocessing import MinMaxScaler
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df[[target_col]])
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, seq_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = Sequential()
        for i in range(self.layers):
            self.model.add(LSTM(self.units, return_sequences=(i < self.layers - 1), input_shape=(seq_length, 1)))
            if self.dropout > 0:
                self.model.add(Dropout(self.dropout))
                
        self.model.add(Dense(1))
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        
        return self
    
    def predict(self, df: pd.DataFrame, target_col: str = "demand", 
                time_col: str = "month_year", seq_length: int = 10) -> np.ndarray:
        """Generate predictions.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            time_col: Time column name
            seq_length: Sequence length
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values(time_col)
        
        scaled_data = self.scaler.transform(df[[target_col]])
        X, _ = self._create_sequences(scaled_data, seq_length)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        predictions = self.model.predict(X, verbose=0)
        return self.scaler.inverse_transform(predictions).flatten()
"""Data Loader Module.

Handles loading data from various sources including CSV, Excel, and databases.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union


class DataLoader:
    """Load data from various sources for the pricing optimization system.
    
    Attributes:
        data_path: Path to the data directory
        file_format: Format of the data file (csv, excel, parquet)
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """Initialize the DataLoader.
        
        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
        
    def load_csv(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            filename: Name of the CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame containing the loaded data
        """
        filepath = self.data_path / filename
        return pd.read_csv(filepath, **kwargs)
    
    def load_excel(self, filename: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from Excel file.
        
        Args:
            filename: Name of the Excel file
            sheet_name: Name of the sheet to load
            **kwargs: Additional arguments for pd.read_excel
            
        Returns:
            DataFrame containing the loaded data
        """
        filepath = self.data_path / filename
        return pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    
    def load_parquet(self, filename: str, **kwargs) -> pd.DataFrame:
        """Load data from Parquet file.
        
        Args:
            filename: Name of the Parquet file
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            DataFrame containing the loaded data
        """
        filepath = self.data_path / filename
        return pd.read_parquet(filepath, **kwargs)
    
    def load_raw_data(self, filename: str = "retail_price.csv") -> pd.DataFrame:
        """Load the main retail price dataset.
        
        Args:
            filename: Name of the data file
            
        Returns:
            DataFrame containing the retail price data
        """
        return self.load_csv(filename)
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """Get information about the loaded data.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing data information
        """
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum()
        }
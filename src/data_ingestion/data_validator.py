"""Data Validator Module.

Validates data quality and schema compliance for the pricing optimization system.
"""

import pandas as pd
from typing import Dict, List, Optional


class DataValidator:
    """Validate data quality and schema compliance.
    
    Attributes:
        schema: Expected schema for validation
    """
    
    def __init__(self, schema: Optional[Dict] = None):
        """Initialize the DataValidator.
        
        Args:
            schema: Expected schema dictionary
        """
        self.schema = schema or {}
        
    def validate_schema(self, df: pd.DataFrame) -> Dict:
        """Validate DataFrame schema.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required columns
        if self.schema.get("required_columns"):
            missing = set(self.schema["required_columns"]) - set(df.columns)
            if missing:
                results["valid"] = False
                results["errors"].append(f"Missing required columns: {missing}")
                
        # Check data types
        if self.schema.get("expected_types"):
            for col, expected_type in self.schema["expected_types"].items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if expected_type not in actual_type:
                        results["warnings"].append(
                            f"Column {col} has type {actual_type}, expected {expected_type}"
                        )
                        
        return results
    
    def validate_ranges(self, df: pd.DataFrame, ranges: Dict) -> Dict:
        """Validate that values are within expected ranges.
        
        Args:
            df: DataFrame to validate
            ranges: Dictionary of column -> (min, max) ranges
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "out_of_range": {}
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    results["valid"] = False
                    results["out_of_range"][col] = len(out_of_range)
                    
        return results
    
    def validate_no_nulls(self, df: pd.DataFrame, columns: List[str]) -> Dict:
        """Validate that specified columns have no null values.
        
        Args:
            df: DataFrame to validate
            columns: Columns that should not have nulls
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            "valid": True,
            "null_columns": {}
        }
        
        for col in columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    results["valid"] = False
                    results["null_columns"][col] = null_count
                    
        return results
    
    def validate_unique(self, df: pd.DataFrame, columns: List[str]) -> Dict:
        """Validate that specified columns have unique values.
        
        Args:
            df: DataFrame to validate
            columns: Columns that should have unique values
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            "valid": True,
            "duplicate_columns": {}
        }
        
        for col in columns:
            if col in df.columns:
                duplicates = df[col].duplicated().sum()
                if duplicates > 0:
                    results["valid"] = False
                    results["duplicate_columns"][col] = duplicates
                    
        return results
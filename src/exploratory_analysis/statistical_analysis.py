"""Statistical Analysis Module.

Performs statistical analysis for the pricing optimization system.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional


class StatisticalAnalysis:
    """Perform statistical analysis for pricing optimization.
    
    Attributes:
        confidence_level: Confidence level for statistical tests
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize the StatisticalAnalysis.
        
        Args:
            confidence_level: Confidence level for statistical tests
        """
        self.confidence_level = confidence_level
        
    def calculate_descriptive_stats(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict:
        """Calculate descriptive statistics.
        
        Args:
            df: DataFrame to analyze
            columns: Columns to include
            
        Returns:
            Dictionary containing descriptive statistics
        """
        if columns:
            df_subset = df[columns]
        else:
            df_subset = df.select_dtypes(include=[np.number])
            
        return {
            "mean": df_subset.mean().to_dict(),
            "median": df_subset.median().to_dict(),
            "std": df_subset.std().to_dict(),
            "min": df_subset.min().to_dict(),
            "max": df_subset.max().to_dict(),
            "q25": df_subset.quantile(0.25).to_dict(),
            "q75": df_subset.quantile(0.75).to_dict()
        }
    
    def test_normality(self, df: pd.DataFrame, column: str) -> Dict:
        """Test if a column follows a normal distribution.
        
        Args:
            df: DataFrame to analyze
            column: Column to test
            
        Returns:
            Dictionary containing test results
        """
        data = df[column].dropna()
        
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(data)
        
        return {
            "test": "Shapiro-Wilk",
            "statistic": stat,
            "p_value": p_value,
            "is_normal": p_value > (1 - self.confidence_level)
        }
    
    def calculate_correlation(self, df: pd.DataFrame, col1: str, col2: str, method: str = "pearson") -> Dict:
        """Calculate correlation between two columns.
        
        Args:
            df: DataFrame to analyze
            col1: First column
            col2: Second column
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Dictionary containing correlation results
        """
        if method == "pearson":
            corr, p_value = stats.pearsonr(df[col1], df[col2])
        elif method == "spearman":
            corr, p_value = stats.spearmanr(df[col1], df[col2])
        else:
            corr, p_value = stats.kendalltau(df[col1], df[col2])
            
        return {
            "correlation": corr,
            "p_value": p_value,
            "method": method,
            "significant": p_value < (1 - self.confidence_level)
        }
    
    def test_hypothesis(self, df: pd.DataFrame, col1: str, col2: str, 
                       test_type: str = "ttest") -> Dict:
        """Test hypothesis about two groups.
        
        Args:
            df: DataFrame to analyze
            col1: First column/group
            col2: Second column/group
            test_type: Type of test (ttest, anova, mannwhitney)
            
        Returns:
            Dictionary containing test results
        """
        if test_type == "ttest":
            stat, p_value = stats.ttest_ind(df[col1], df[col2])
        elif test_type == "anova":
            stat, p_value = stats.f_oneway(*[df[c] for c in [col1, col2]])
        else:
            stat, p_value = stats.mannwhitneyu(df[col1], df[col2])
            
        return {
            "test": test_type,
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < (1 - self.confidence_level)
        }
    
    def calculate_confidence_interval(self, df: pd.DataFrame, column: str) -> Dict:
        """Calculate confidence interval for a column.
        
        Args:
            df: DataFrame to analyze
            column: Column to analyze
            
        Returns:
            Dictionary containing confidence interval
        """
        data = df[column].dropna()
        mean = data.mean()
        sem = stats.sem(data)
        
        ci = stats.t.interval(
            self.confidence_level, 
            len(data) - 1, 
            loc=mean, 
            scale=sem
        )
        
        return {
            "mean": mean,
            "lower_bound": ci[0],
            "upper_bound": ci[1],
            "confidence_level": self.confidence_level
        }
    
    def detect_outliers(self, df: pd.DataFrame, column: str, method: str = "iqr") -> Dict:
        """Detect outliers in a column.
        
        Args:
            df: DataFrame to analyze
            column: Column to analyze
            method: Detection method (iqr, zscore)
            
        Returns:
            Dictionary containing outlier information
        """
        data = df[column].dropna()
        
        if method == "iqr":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = data[(data < lower) | (data > upper)]
        else:
            z_scores = np.abs(stats.zscore(data))
            outliers = data[z_scores > 3]
            
        return {
            "method": method,
            "outlier_count": len(outliers),
            "outlier_percentage": (len(outliers) / len(data)) * 100,
            "outlier_values": outliers.tolist()[:10]  # First 10 outliers
        }
"""Visualizations Module.

Creates various visualizations for the pricing optimization analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple


class Visualizations:
    """Create visualizations for pricing optimization analysis.
    
    Attributes:
        style: Matplotlib style to use
        color_palette: Seaborn color palette
    """
    
    def __init__(self, style: str = "whitegrid", color_palette: str = "husl"):
        """Initialize the Visualizations.
        
        Args:
            style: Matplotlib style to use
            color_palette: Seaborn color palette
        """
        sns.set_style(style)
        self.color_palette = color_palette
        
    def plot_demand_trend(self, df: pd.DataFrame, time_col: str = "month_year", 
                         demand_col: str = "demand", category_col: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot demand trend over time.
        
        Args:
            df: DataFrame to plot
            time_col: Time column
            demand_col: Demand column
            category_col: Category column for grouping
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if category_col and category_col in df.columns:
            for category in df[category_col].unique():
                subset = df[df[category_col] == category]
                ax.plot(subset[time_col], subset[demand_col], label=category)
            ax.legend()
        else:
            ax.plot(df[time_col], df[demand_col])
            
        ax.set_xlabel("Time")
        ax.set_ylabel("Demand")
        ax.set_title("Demand Trend Over Time")
        plt.xticks(rotation=45)
        
        return fig
    
    def plot_price_distribution(self, df: pd.DataFrame, price_col: str = "avg_price",
                                 category_col: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot price distribution.
        
        Args:
            df: DataFrame to plot
            price_col: Price column
            category_col: Category column for grouping
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if category_col and category_col in df.columns:
            for category in df[category_col].unique():
                subset = df[df[category_col] == category][price_col]
                ax.hist(subset, alpha=0.5, label=category, bins=30)
            ax.legend()
        else:
            ax.hist(df[price_col], bins=30)
            
        ax.set_xlabel("Price")
        ax.set_ylabel("Frequency")
        ax.set_title("Price Distribution")
        
        return fig
    
    def plot_correlation_matrix(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
        """Plot correlation matrix heatmap.
        
        Args:
            df: DataFrame to plot
            columns: Columns to include
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if columns:
            df_subset = df[columns]
        else:
            df_subset = df.select_dtypes(include=[np.number])
            
        fig, ax = plt.subplots(figsize=figsize)
        corr = df_subset.corr()
        
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", 
                    center=0, ax=ax, square=True)
        ax.set_title("Correlation Matrix")
        
        return fig
    
    def plot_price_elasticity_curve(self, prices: np.ndarray, demands: np.ndarray,
                                     figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """Plot price elasticity curve.
        
        Args:
            prices: Array of prices
            demands: Array of corresponding demands
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(prices, demands, marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("Price")
        ax.set_ylabel("Demand")
        ax.set_title("Price-Demand Relationship (Elasticity Curve)")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_category_performance(self, df: pd.DataFrame, category_col: str = "category",
                                  demand_col: str = "demand", price_col: str = "avg_price",
                                  figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """Plot category performance comparison.
        
        Args:
            df: DataFrame to plot
            category_col: Category column
            demand_col: Demand column
            price_col: Price column
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Demand by category
        category_demand = df.groupby(category_col)[demand_col].sum().sort_values(ascending=False)
        ax1.bar(range(len(category_demand)), category_demand.values)
        ax1.set_xticks(range(len(category_demand)))
        ax1.set_xticklabels(category_demand.index, rotation=45, ha="right")
        ax1.set_ylabel("Total Demand")
        ax1.set_title("Total Demand by Category")
        
        # Average price by category
        category_price = df.groupby(category_col)[price_col].mean().sort_values(ascending=False)
        ax2.bar(range(len(category_price)), category_price.values)
        ax2.set_xticks(range(len(category_price)))
        ax2.set_xticklabels(category_price.index, rotation=45, ha="right")
        ax2.set_ylabel("Average Price")
        ax2.set_title("Average Price by Category")
        
        plt.tight_layout()
        return fig
    
    def plot_seasonality(self, df: pd.DataFrame, time_col: str = "month_year",
                        demand_col: str = "demand", figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot seasonal patterns.
        
        Args:
            df: DataFrame to plot
            time_col: Time column
            demand_col: Demand column
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        df_temp = df.copy()
        df_temp[time_col] = pd.to_datetime(df_temp[time_col])
        df_temp["month"] = df_temp[time_col].dt.month
        
        fig, ax = plt.subplots(figsize=figsize)
        
        monthly_demand = df_temp.groupby("month")[demand_col].mean()
        ax.bar(monthly_demand.index, monthly_demand.values, color="steelblue")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average Demand")
        ax.set_title("Seasonal Demand Pattern")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        
        return fig
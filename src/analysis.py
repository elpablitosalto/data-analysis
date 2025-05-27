import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def calculate_basic_stats(df: pd.DataFrame) -> Dict:
    """
    Calculate basic statistical measures for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict: Dictionary containing statistical measures
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats = {}
    
    for col in numeric_cols:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'q1': df[col].quantile(0.25),
            'q3': df[col].quantile(0.75)
        }
    
    return stats

def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Correlation matrix
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].corr()

def analyze_categorical(df: pd.DataFrame) -> Dict:
    """
    Analyze categorical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict: Dictionary containing categorical analysis results
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    analysis = {}
    
    for col in categorical_cols:
        analysis[col] = {
            'unique_values': df[col].nunique(),
            'value_counts': df[col].value_counts().to_dict(),
            'most_common': df[col].mode()[0]
        }
    
    return analysis

def detect_outliers(df: pd.DataFrame, columns: List[str] = None) -> Dict:
    """
    Detect outliers in numeric columns using IQR method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str], optional): List of columns to analyze. If None, all numeric columns are used.
        
    Returns:
        Dict: Dictionary containing outlier information
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers[col] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_count': len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
        }
    
    return outliers 
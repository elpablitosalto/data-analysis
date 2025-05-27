import pandas as pd
import numpy as np
from typing import Optional, Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Make a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    for column in df_clean.columns:
        if df_clean[column].dtype in ['int64', 'float64']:
            # For numeric columns, fill with median
            df_clean[column].fillna(df_clean[column].median(), inplace=True)
        else:
            # For categorical columns, fill with mode
            df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
    
    # Remove duplicates
    df_clean.drop_duplicates(inplace=True)
    
    return df_clean

def get_data_info(df: pd.DataFrame) -> dict:
    """
    Get basic information about the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return info 
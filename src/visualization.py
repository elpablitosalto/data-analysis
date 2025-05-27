import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Optional
import os

def setup_plot_style():
    """Set up the style for all plots."""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def save_plot(filename: str):
    """
    Save the current plot to the reports directory.
    
    Args:
        filename (str): Name of the file to save
    """
    plt.tight_layout()
    plt.savefig(os.path.join('reports', filename))
    plt.close()

def plot_numeric_distributions(df: pd.DataFrame, columns: Optional[List[str]] = None):
    """
    Create distribution plots for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str], optional): List of columns to plot. If None, all numeric columns are used.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    setup_plot_style()
    
    for col in columns:
        plt.figure()
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        save_plot(f'distribution_{col}.png')

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
    """
    setup_plot_style()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    save_plot('correlation_heatmap.png')

def plot_categorical_counts(df: pd.DataFrame, columns: Optional[List[str]] = None):
    """
    Create count plots for categorical columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str], optional): List of columns to plot. If None, all categorical columns are used.
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns
    
    setup_plot_style()
    
    for col in columns:
        plt.figure()
        sns.countplot(data=df, x=col)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        save_plot(f'count_{col}.png')

def plot_boxplots(df: pd.DataFrame, columns: Optional[List[str]] = None):
    """
    Create boxplots for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str], optional): List of columns to plot. If None, all numeric columns are used.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    setup_plot_style()
    
    for col in columns:
        plt.figure()
        sns.boxplot(data=df, y=col)
        plt.title(f'Boxplot of {col}')
        save_plot(f'boxplot_{col}.png')

def plot_scatter_matrix(df: pd.DataFrame, columns: Optional[List[str]] = None):
    """
    Create a scatter plot matrix for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        columns (List[str], optional): List of columns to plot. If None, all numeric columns are used.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    setup_plot_style()
    
    plt.figure()
    sns.pairplot(df[columns])
    save_plot('scatter_matrix.png') 
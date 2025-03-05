"""
Utility functions for data loading and analysis.
"""
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

def load_loan_data(csv_path: Path) -> pd.DataFrame:
    """
    Load loan data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the loan data.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=['date'])
        print(f"Loaded {len(df)} rows from {csv_path}")
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data from {csv_path}: {str(e)}")

def generate_descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate descriptive statistics for the loan data.
    
    Args:
        df: DataFrame containing the loan data.
        
    Returns:
        Dictionary with descriptive statistics.
    """
    stats = {
        "total_loans": len(df),
        "date_range": {
            "start": df['date'].min().strftime('%Y-%m-%d'),
            "end": df['date'].max().strftime('%Y-%m-%d'),
            "years_covered": (df['date'].max().year - df['date'].min().year) + 1
        },
        "loans_by_year": df.groupby(df['date'].dt.year).size().to_dict(),
        "unique_lead_arrangers": df['lead_arranger'].nunique(),
        "unique_lenders": df['lender_name'].nunique(),
        "unique_borrowers": df['borrower_name'].nunique()
    }
    
    return stats

class NumpyEncoder(json.JSONEncoder):
    """Special JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_stats_to_json(stats: Dict[str, Any], output_path: Path) -> None:
    """
    Save statistics to a JSON file.
    
    Args:
        stats: Dictionary of statistics.
        output_path: Path where to save the JSON file.
    """
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, cls=NumpyEncoder)
    print(f"Statistics saved to {output_path}")

def plot_loans_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create a plot showing the number of loans over time.
    
    Args:
        df: DataFrame containing the loan data.
        output_dir: Directory to save the plot.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot loans count by year
    yearly_counts = df.groupby(df['date'].dt.year).size()
    ax = yearly_counts.plot(kind='bar', color='steelblue')
    
    plt.title('Number of Loans by Year', fontsize=14)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Loans', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    output_path = output_dir / "loans_by_year.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}") 
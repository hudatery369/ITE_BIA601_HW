"""
Data Loader Module - Simplified Version
Load datasets from CSV or Excel files
"""

import pandas as pd
from pathlib import Path


class DataLoader:
    """
    Simple Data Loader for CSV and Excel files
    """
    
    def __init__(self):
        """Initialize DataLoader"""
        self.supported_formats = ['.csv', '.xlsx', '.xls']
    
    def load_file(self, filepath):
        """
        Load data from file (auto-detect format)
        
        Args:
            filepath: Path to file
            
        Returns:
            pandas DataFrame
        """
        path = Path(filepath)
        
        # Check file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Get extension
        ext = path.suffix.lower()
        
        # Check format supported
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported format: {ext}")
        
        # Load based on extension
        try:
            if ext == '.csv':
                df = pd.read_csv(filepath)
            else:  # Excel
                df = pd.read_excel(filepath)
            
            print(f"✓ Loaded: {path.name}")
            print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            return df
            
        except Exception as e:
            raise Exception(f"Error loading file: {str(e)}")
    
    def get_info(self, df):
        """
        Get basic dataset information
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict with dataset info
        """
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'features': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing': df.isnull().sum().to_dict()
        }

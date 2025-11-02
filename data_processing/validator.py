"""
Data Validator Module - Simplified Version
Basic validation for datasets
"""

import pandas as pd
import numpy as np


class DataValidator:
    """
    Simple Data Validator
    Check basic data quality
    """
    
    def __init__(self):
        """Initialize validator"""
        pass
    
    def validate_dataset(self, df, target_column):
        """
        Validate dataset for GA feature selection
        
        Args:
            df: pandas DataFrame
            target_column: Target column name
            
        Returns:
            bool: True if valid, False otherwise
        """
        print("\n" + "="*60)
        print("VALIDATING DATASET")
        print("="*60)
        
        errors = []
        
        # Check 1: Not empty
        if df.empty:
            errors.append("Dataset is empty")
        
        # Check 2: Minimum columns (at least 2: 1 feature + 1 target)
        if len(df.columns) < 2:
            errors.append("Need at least 2 columns (features + target)")
        
        # Check 3: Minimum rows
        if len(df) < 10:
            errors.append("Need at least 10 samples")
        
        # Check 4: Target column exists
        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found")
        else:
            # Check 5: Target has at least 2 classes
            n_classes = df[target_column].nunique()
            if n_classes < 2:
                errors.append(f"Target needs at least 2 classes (found {n_classes})")
        
        # Print results
        if errors:
            print("✗ VALIDATION FAILED:")
            for error in errors:
                print(f"  - {error}")
            print("="*60)
            return False
        else:
            print("✓ VALIDATION PASSED")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            if target_column in df.columns:
                print(f"  Classes: {df[target_column].nunique()}")
            print("="*60)
            return True
    
    def check_missing_values(self, df):
        """
        Check for missing values
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict with missing value info
        """
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        pct = (total_missing / total_cells) * 100
        
        return {
            'total_missing': total_missing,
            'percentage': pct,
            'has_missing': total_missing > 0
        }

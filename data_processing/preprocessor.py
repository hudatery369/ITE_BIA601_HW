"""
Data Preprocessor Module - Simplified Version
Handle missing values, encoding, and normalization
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    Simple Data Preprocessor
    Handles common preprocessing tasks
    """
    
    def __init__(self):
        """Initialize preprocessor"""
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def handle_missing_values(self, df, strategy='mean'):
        """
        Fill missing values
        
        Args:
            df: DataFrame
            strategy: 'mean' or 'median' or 'mode'
            
        Returns:
            DataFrame with no missing values
        """
        df_clean = df.copy()
        
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['float64', 'int64']:
                    # Numeric columns
                    if strategy == 'mean':
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                else:
                    # Categorical columns
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        return df_clean
    
    def encode_categorical(self, df):
        """
        Encode categorical features using Label Encoder
        
        Args:
            df: DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object':
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = encoder
        
        return df_encoded
    
    def separate_features_target(self, df, target_column):
        """
        Separate features (X) and target (y)
        
        Args:
            df: DataFrame
            target_column: Name of target column
            
        Returns:
            X (features), y (target)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        return X, y
    
    def normalize_features(self, X, method='standard'):
        """
        Normalize features
        
        Args:
            X: Feature matrix (numpy array or DataFrame)
            method: 'standard' (mean=0, std=1)
            
        Returns:
            Normalized feature matrix
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if method == 'standard':
            X_normalized = self.scaler.fit_transform(X)
        else:
            X_normalized = X
        
        return X_normalized
    
    def full_preprocessing(self, df, target_column):
        """
        Complete preprocessing pipeline
        
        Args:
            df: Raw DataFrame
            target_column: Target column name
            
        Returns:
            X_normalized (features), y (target), feature_names
        """
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        
        # 1. Handle missing values
        print("1. Handling missing values...")
        df_clean = self.handle_missing_values(df)
        
        # 2. Encode categorical features
        print("2. Encoding categorical features...")
        df_encoded = self.encode_categorical(df_clean)
        
        # 3. Separate features and target
        print("3. Separating features and target...")
        X, y = self.separate_features_target(df_encoded, target_column)
        feature_names = X.columns.tolist()
        
        # 4. Normalize features
        print("4. Normalizing features...")
        X_normalized = self.normalize_features(X.values)
        
        print(f"\n✓ Preprocessing complete!")
        print(f"  Features: {X_normalized.shape[1]}")
        print(f"  Samples: {X_normalized.shape[0]}")
        print(f"  Target classes: {len(np.unique(y))}")
        print("="*60)
        
        return X_normalized, y.values, feature_names

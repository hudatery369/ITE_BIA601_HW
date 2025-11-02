"""
Machine Learning Models Module - Simplified Version
Simple interface for training and evaluating models
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class MLModel:
    """
    Simple ML Model wrapper
    Uses Random Forest by default
    """
    
    def __init__(self, random_state=42):
        """
        Initialize ML Model
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
    
    def train(self, X_train, y_train):
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Trained model
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        train_acc = self.model.score(X_train, y_train)
        
        print(f"✓ Model trained (Training accuracy: {train_acc:.4f})")
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Accuracy score
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        accuracy = self.model.score(X_test, y_test)
        return accuracy
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Target vector
            cv: Number of folds
            
        Returns:
            dict with CV scores
        """
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'scores': scores,
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importances (for Random Forest)
        
        Args:
            feature_names: List of feature names
            
        Returns:
            dict with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Create sorted dictionary
        importance_dict = dict(zip(feature_names, importances))
        sorted_importance = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return sorted_importance

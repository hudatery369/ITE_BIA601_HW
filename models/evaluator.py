"""
Model Evaluator Module - Simplified Version
Evaluate feature subsets using ML models
"""

import numpy as np
import time
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class ModelEvaluator:
    """
    Simple Model Evaluator
    Evaluates feature subsets using Random Forest
    """
    
    def __init__(self, random_state=42):
        """Initialize evaluator"""
        self.random_state = random_state
    
    def evaluate_features(self, X, y, cv=3):
        """
        Evaluate feature subset using cross-validation
        
        Args:
            X: Feature matrix (only selected features)
            y: Target vector
            cv: Number of CV folds
            
        Returns:
            dict with evaluation metrics
        """
        start_time = time.time()
        
        # Create model
        model = RandomForestClassifier(
            n_estimators=50,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Train on all data for other metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred,
            average='weighted',
            zero_division=0
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            'accuracy': np.mean(cv_scores),
            'accuracy_std': np.std(cv_scores),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'time': elapsed_time
        }
    
    def compare_feature_subsets(self, X, y, feature_subsets, subset_names=None):
        """
        Compare multiple feature subsets
        
        Args:
            X: Full feature matrix
            y: Target vector
            feature_subsets: List of feature index arrays
            subset_names: Optional names for subsets
            
        Returns:
            dict with comparison results
        """
        if subset_names is None:
            subset_names = [f"Subset_{i+1}" for i in range(len(feature_subsets))]
        
        print("\n" + "="*60)
        print("COMPARING FEATURE SUBSETS")
        print("="*60)
        
        results = {}
        
        for name, indices in zip(subset_names, feature_subsets):
            print(f"\nEvaluating {name} ({len(indices)} features)...")
            
            X_subset = X[:, indices]
            metrics = self.evaluate_features(X_subset, y)
            
            results[name] = {
                'n_features': len(indices),
                'indices': indices.tolist() if hasattr(indices, 'tolist') else list(indices),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'time': metrics['time']
            }
            
            print(f"  Accuracy: {metrics['accuracy']:.4f} ± {metrics['accuracy_std']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        print("\n" + "="*60)
        print("COMPARISON COMPLETE")
        print("="*60)
        
        # Sort by accuracy
        sorted_results = dict(sorted(
            results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        ))
        
        print("\nRanking by Accuracy:")
        for i, (name, res) in enumerate(sorted_results.items()):
            print(f"{i+1}. {name:20s} - Accuracy: {res['accuracy']:.4f} ({res['n_features']} features)")
        
        return sorted_results

"""
Fitness Evaluation Module - Simplified Version
Evaluates chromosomes based on accuracy and feature count
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


class FitnessEvaluator:
    """
    Fitness Evaluator: Calculates fitness score for chromosomes
    Fitness = (alpha * accuracy) + ((1 - alpha) * feature_reduction)
    """
    
    def __init__(self, X, y, alpha=0.9):
        """
        Initialize fitness evaluator
        
        Args:
            X: Feature matrix (samples x features)
            y: Target vector
            alpha: Weight for accuracy (higher = prioritize accuracy)
        """
        self.X = X
        self.y = y
        self.alpha = alpha
        self.n_features = X.shape[1]
        
        # Split data once for fast evaluation
        # Try stratified split, fall back to regular split if not possible
        try:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError:
            # If stratify fails (too few samples per class), use regular split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
    
    def evaluate(self, chromosome):
        """
        Evaluate fitness of a chromosome
        
        Args:
            chromosome: Chromosome to evaluate
            
        Returns:
            fitness score (0 to 1, higher is better)
        """
        # Get selected features
        selected = chromosome.get_selected_features()
        n_selected = len(selected)
        
        # If no features selected, return 0
        if n_selected == 0:
            chromosome.fitness = 0.0
            chromosome.accuracy = 0.0
            return 0.0
        
        # Train and evaluate model
        try:
            X_train_selected = self.X_train[:, selected]
            X_test_selected = self.X_test[:, selected]
            
            # Use Random Forest classifier
            model = RandomForestClassifier(
                n_estimators=50, 
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_selected, self.y_train)
            accuracy = model.score(X_test_selected, self.y_test)
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            accuracy = 0.0
        
        # Calculate feature reduction ratio
        feature_reduction = 1.0 - (n_selected / self.n_features)
        
        # Calculate fitness (weighted combination)
        fitness = (self.alpha * accuracy) + ((1 - self.alpha) * feature_reduction)
        
        # Update chromosome
        chromosome.fitness = fitness
        chromosome.accuracy = accuracy
        
        return fitness
    
    def evaluate_population(self, population):
        """
        Evaluate all chromosomes in population
        
        Args:
            population: List of chromosomes
        """
        for chromosome in population:
            self.evaluate(chromosome)
    
    def get_best(self, population):
        """Get chromosome with highest fitness"""
        return max(population, key=lambda c: c.fitness)
    
    def get_stats(self, population):
        """
        Get population statistics
        
        Returns:
            dict with best, average, worst fitness
        """
        fitness_values = [c.fitness for c in population]
        accuracies = [c.accuracy for c in population]
        features_counts = [c.count_selected() for c in population]
        
        return {
            'best_fitness': np.max(fitness_values),
            'avg_fitness': np.mean(fitness_values),
            'worst_fitness': np.min(fitness_values),
            'best_accuracy': np.max(accuracies),
            'avg_features': np.mean(features_counts)
        }

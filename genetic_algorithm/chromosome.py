"""
Chromosome Module - Simplified Version
Represents a solution (feature subset) in the genetic algorithm
"""

import numpy as np
import random


class Chromosome:
    """
    Chromosome: Binary encoding for feature selection
    Each gene (bit): 1 = feature selected, 0 = feature not selected
    """
    
    def __init__(self, n_features, genes=None):
        """
        Initialize chromosome
        
        Args:
            n_features: Total number of features
            genes: Binary array (optional, if None creates random)
        """
        self.n_features = n_features
        
        if genes is not None:
            self.genes = np.array(genes, dtype=int)
        else:
            # Create random chromosome with at least 1 feature selected
            self.genes = np.random.randint(0, 2, size=n_features)
            while self.genes.sum() == 0:
                self.genes = np.random.randint(0, 2, size=n_features)
        
        # Initialize fitness metrics
        self.fitness = 0.0
        self.accuracy = 0.0
    
    def get_selected_features(self):
        """Get indices of selected features"""
        return np.where(self.genes == 1)[0]
    
    def count_selected(self):
        """Count how many features are selected"""
        return int(self.genes.sum())
    
    def flip_gene(self, index):
        """Flip gene at index (for mutation)"""
        self.genes[index] = 1 - self.genes[index]
    
    def copy(self):
        """Create a copy of this chromosome"""
        new_chrom = Chromosome(self.n_features, genes=self.genes.copy())
        new_chrom.fitness = self.fitness
        new_chrom.accuracy = self.accuracy
        return new_chrom
    
    def __str__(self):
        """String representation"""
        return f"Chromosome(fitness={self.fitness:.4f}, features={self.count_selected()}/{self.n_features})"
    
    @staticmethod
    def create_population(pop_size, n_features):
        """
        Create initial random population
        
        Args:
            pop_size: Population size
            n_features: Number of features
            
        Returns:
            List of random chromosomes
        """
        return [Chromosome(n_features) for _ in range(pop_size)]


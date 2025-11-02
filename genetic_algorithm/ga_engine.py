"""
Genetic Algorithm Engine - Simplified Version
Main engine for feature selection using genetic algorithm
"""

import numpy as np
import time
from .chromosome import Chromosome
from .fitness import FitnessEvaluator
from .operators import GeneticOperators


class GeneticAlgorithm:
    """
    Genetic Algorithm for Feature Selection
    Simplified version with essential parameters only
    """
    
    def __init__(self, 
                 population_size=50,
                 n_generations=100,
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 alpha=0.9,
                 verbose=True):
        """
        Initialize Genetic Algorithm
        
        Args:
            population_size: Number of chromosomes in population
            n_generations: Number of generations to evolve
            crossover_rate: Probability of crossover (0-1)
            mutation_rate: Probability of mutation per gene (0-1)
            alpha: Weight for accuracy vs feature reduction (0-1)
            verbose: Print progress information
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.alpha = alpha
        self.verbose = verbose
        
        # Number of elite chromosomes (top 10%)
        self.n_elite = max(1, int(population_size * 0.1))
        
        # Initialize operators
        self.operators = GeneticOperators(crossover_rate, mutation_rate)
        
        # Track best solution and history
        self.best_chromosome = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_accuracy': [],
            'avg_features': []
        }
    
    def fit(self, X, y):
        """
        Run genetic algorithm on dataset
        
        Args:
            X: Feature matrix (samples x features)
            y: Target vector
            
        Returns:
            Best chromosome found
        """
        start_time = time.time()
        
        if self.verbose:
            print("\n" + "="*70)
            print("GENETIC ALGORITHM FOR FEATURE SELECTION")
            print("="*70)
            print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.n_generations}")
            print(f"Crossover rate: {self.crossover_rate}")
            print(f"Mutation rate: {self.mutation_rate}")
            print("="*70)
        
        # Initialize fitness evaluator
        evaluator = FitnessEvaluator(X, y, alpha=self.alpha)
        
        # Create initial population
        if self.verbose:
            print("\nInitializing random population...")
        population = Chromosome.create_population(self.population_size, X.shape[1])
        
        # Evaluate initial population
        evaluator.evaluate_population(population)
        self._update_history(population, evaluator, 0)
        
        # Evolution loop
        if self.verbose:
            print(f"\n{'='*70}")
            print("EVOLUTION STARTED")
            print(f"{'='*70}\n")
        
        for generation in range(1, self.n_generations + 1):
            # Create new generation
            population = self._evolve_generation(population)
            
            # Evaluate new population
            evaluator.evaluate_population(population)
            
            # Update history and best solution
            self._update_history(population, evaluator, generation)
        
        # Final results
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*70}")
            print("EVOLUTION COMPLETED")
            print(f"{'='*70}")
            print(f"Time elapsed: {elapsed_time:.2f} seconds")
            print(f"\nBest Solution:")
            print(f"  Fitness: {self.best_chromosome.fitness:.4f}")
            print(f"  Accuracy: {self.best_chromosome.accuracy:.4f}")
            print(f"  Features selected: {self.best_chromosome.count_selected()}/{X.shape[1]}")
            print(f"  Feature reduction: {(1 - self.best_chromosome.count_selected()/X.shape[1])*100:.1f}%")
            print(f"{'='*70}\n")
        
        return self.best_chromosome
    
    def _evolve_generation(self, population):
        """
        Evolve one generation
        
        Args:
            population: Current population
            
        Returns:
            New population
        """
        # Keep elite chromosomes
        new_population = self.operators.elitism(population, self.n_elite)
        
        # Fill rest with offspring
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            parent1 = self.operators.tournament_selection(population)
            parent2 = self.operators.tournament_selection(population)
            
            # Crossover
            child1, child2 = self.operators.single_point_crossover(parent1, parent2)
            
            # Mutation
            child1 = self.operators.bit_flip_mutation(child1)
            child2 = self.operators.bit_flip_mutation(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population[:self.population_size]
    
    def _update_history(self, population, evaluator, generation):
        """
        Update history with current generation statistics
        
        Args:
            population: Current population
            evaluator: Fitness evaluator
            generation: Current generation number
        """
        stats = evaluator.get_stats(population)
        
        self.history['best_fitness'].append(stats['best_fitness'])
        self.history['avg_fitness'].append(stats['avg_fitness'])
        self.history['best_accuracy'].append(stats['best_accuracy'])
        self.history['avg_features'].append(stats['avg_features'])
        
        # Update best chromosome
        best = evaluator.get_best(population)
        if self.best_chromosome is None or best.fitness > self.best_chromosome.fitness:
            self.best_chromosome = best.copy()
        
        # Print progress
        if self.verbose:
            print(f"Gen {generation:3d} | "
                  f"Best Fitness: {stats['best_fitness']:.4f} | "
                  f"Avg Fitness: {stats['avg_fitness']:.4f} | "
                  f"Accuracy: {stats['best_accuracy']:.4f} | "
                  f"Avg Features: {stats['avg_features']:.1f}")
    
    def get_selected_features(self):
        """Get indices of selected features from best chromosome"""
        if self.best_chromosome is None:
            raise ValueError("GA has not been run yet. Call fit() first.")
        return self.best_chromosome.get_selected_features()
    
    def get_history(self):
        """Get evolution history"""
        return self.history


# Example usage
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    
    # Load dataset
    X, y = load_iris(return_X_y=True)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features\n")
    
    # Create and run GA
    ga = GeneticAlgorithm(
        population_size=20,
        n_generations=30,
        crossover_rate=0.8,
        mutation_rate=0.1,
        alpha=0.9,
        verbose=True
    )
    
    # Run GA
    best_solution = ga.fit(X, y)
    
    # Get selected features
    selected = ga.get_selected_features()
    print(f"\nSelected features: {selected}")

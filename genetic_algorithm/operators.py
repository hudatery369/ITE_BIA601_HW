"""
Genetic Operators Module - Simplified Version
Implements selection, crossover, and mutation
"""

import numpy as np
import random
from .chromosome import Chromosome


class GeneticOperators:
    """
    Genetic Operators: Selection, Crossover, Mutation
    Simplified to use only the most effective methods
    """
    
    def __init__(self, crossover_rate=0.8, mutation_rate=0.1):
        """
        Initialize genetic operators
        
        Args:
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation per gene
        """
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
    
    # ==================== SELECTION ====================
    
    def tournament_selection(self, population, tournament_size=3):
        """
        Tournament Selection: Pick best from random sample
        
        Args:
            population: List of chromosomes
            tournament_size: Number of chromosomes in tournament
            
        Returns:
            Selected chromosome
        """
        tournament = random.sample(population, tournament_size)
        winner = max(tournament, key=lambda c: c.fitness)
        return winner.copy()
    
    def elitism(self, population, n_elite):
        """
        Elitism: Keep top N chromosomes
        
        Args:
            population: List of chromosomes
            n_elite: Number of elite chromosomes
            
        Returns:
            List of top N chromosomes
        """
        sorted_pop = sorted(population, key=lambda c: c.fitness, reverse=True)
        return [c.copy() for c in sorted_pop[:n_elite]]
    
    # ==================== CROSSOVER ====================
    
    def single_point_crossover(self, parent1, parent2):
        """
        Single-Point Crossover: Exchange genes at one point
        
        Args:
            parent1, parent2: Parent chromosomes
            
        Returns:
            Two offspring chromosomes
        """
        # Apply crossover with probability
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Select random crossover point
        n_genes = parent1.n_features
        point = random.randint(1, n_genes - 1)
        
        # Create offspring
        child1_genes = np.concatenate([
            parent1.genes[:point],
            parent2.genes[point:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:point],
            parent1.genes[point:]
        ])
        
        # Ensure at least one feature selected
        if child1_genes.sum() == 0:
            child1_genes[random.randint(0, n_genes-1)] = 1
        if child2_genes.sum() == 0:
            child2_genes[random.randint(0, n_genes-1)] = 1
        
        child1 = Chromosome(n_genes, genes=child1_genes)
        child2 = Chromosome(n_genes, genes=child2_genes)
        
        return child1, child2
    
    # ==================== MUTATION ====================
    
    def bit_flip_mutation(self, chromosome):
        """
        Bit-Flip Mutation: Flip each gene with mutation probability
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        # Flip each gene with mutation_rate probability
        for i in range(mutated.n_features):
            if random.random() < self.mutation_rate:
                mutated.flip_gene(i)
        
        # Ensure at least one feature selected
        if mutated.genes.sum() == 0:
            mutated.genes[random.randint(0, mutated.n_features - 1)] = 1
        
        return mutated

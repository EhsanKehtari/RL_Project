import random

import numpy as np
from numpy import ndarray


class Selection:
    """
    Implement well-known selection techniques in Genetic Algorithm to be used for flowshop problem.
    List of selection techniques (total=5):
    - random_s
    - greedy_s
    - cost_weighted_roulette_wheel_s
    - rank_weighted_roulette_wheel_s
    - tournament_s
    """

    def __init__(
            self,
            population: ndarray = None,
            objective_function_values: ndarray = None,
            selection_size: int = None
    ):
        """
        :param selection_size: Number of chromosomes to be selected from population.
        :param objective_function_values: Objective function values of each of the chromosomes in the population.
        :param population: Initial population to be selected from.
        """
        self.population = population
        self.selection_size = selection_size
        self.objective_function_values = objective_function_values

    def random_s(self) -> ndarray:
        """
        Select selection_size random chromosomes from the given population.
        :return: array of selected chromosomes
        """
        return np.random.permutation(self.population)[: self.selection_size, :]

    def greedy_s(self) -> ndarray:
        """
        Select selection_size chromosomes from the given population greedily (i.e., only the best selection_size
        chromosomes).
        :return: array of selected chromosomes
        """
        return self.population[np.argsort(self.objective_function_values)][: self.selection_size, :]

    def cost_weighted_roulette_wheel_s(self) -> ndarray:
        """
        Select selection_size chromosomes from the given population based on cost-weighted roulette wheel selection.
        :return: array of selected chromosomes
        """
        # Creating empty subpopulation
        sub_population = np.zeros(
            shape=(self.selection_size, self.population.shape[1]),
            dtype=int
        )
        # Sort the population based on values
        population_sorted = self.population[np.argsort(self.objective_function_values)]
        # Sort values
        objective_function_values_sorted = np.sort(self.objective_function_values)
        # For minimization problem
        fitness_values = 1 / objective_function_values_sorted
        # Normalizing fitness_values
        fitness_values_normalized = fitness_values / np.sum(fitness_values)
        # Selection logic
        for selection_num in range(self.selection_size):
            # Random value for selection
            random_value = random.random()
            # Comparison
            base_idx = 0
            base_value = fitness_values_normalized[base_idx]
            while base_value < random_value:
                base_idx += 1
                base_value += fitness_values_normalized[base_idx]
            # Identify selected chromosome and put it in sub_population
            sub_population[selection_num] = population_sorted[base_idx]
        return sub_population

    def rank_weighted_roulette_wheel_s(self) -> ndarray:
        """
        Select selection_size chromosomes from the given population based on rank-weighted roulette wheel selection.
        :return: array of selected chromosomes
        """
        # Rank values
        rank_values = np.zeros_like(self.objective_function_values)
        denominator = (len(rank_values) * (len(rank_values) + 1)) / 2
        for rank in range(len(rank_values)):
            numerator = len(rank_values) - rank
            rank_values[rank] = numerator / denominator
        # Creating empty subpopulation
        sub_population = np.zeros(
            shape=(self.selection_size, self.population.shape[1]),
            dtype=int
        )
        # Sort the population based on values
        population_sorted = self.population[np.argsort(self.objective_function_values)]
        # Selection logic
        for selection_num in range(self.selection_size):
            # Random value for selection
            random_value = random.random()
            # Comparison
            base_idx = 0
            base_value = rank_values[base_idx]
            while base_value < random_value:
                base_idx += 1
                base_value += rank_values[base_idx]
            # Identify selected chromosome and put it in sub_population
            sub_population[selection_num] = population_sorted[base_idx]
        return sub_population

    def tournament_s(self, subset_size=3):
        """
        Select selection_size chromosomes from the given population based on tournament selection.
        :param subset_size: number of chromosomes to be randomly selected in each round
        :return: array of selected chromosomes
        """
        # Creating empty subpopulation
        sub_population = np.zeros(
            shape=(self.selection_size, self.population.shape[1]),
            dtype=int
        )
        # Selection logic
        for selection_num in range(self.selection_size):
            # Random selection of a subset of chromosomes
            random_subset_idx = np.random.permutation(self.population.shape[0])[:subset_size]
            chromosomes_subset = self.population[random_subset_idx]
            # For minimization problem
            best_local_chromosome = chromosomes_subset[np.argmin(self.objective_function_values[random_subset_idx])]
            # Populating subpopulation
            sub_population[selection_num] = best_local_chromosome
        return sub_population


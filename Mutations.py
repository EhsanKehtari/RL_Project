import inspect

import numpy as np
from numpy import ndarray


class Mutation:
    """
    Implement well-known mutation operators in Genetic Algorithm to be used for flowshop problem.
    List of mutation operators (total=5):
    - insertion_m
    - swap_m
    - inversion_m
    - adjacent_m
    - three_jobs_change_m
    """

    def __init__(self, chromosome: ndarray = None):
        self.chromosome = chromosome
        self.mutated_chromosome = self.chromosome

    def insertion_m(self) -> ndarray:
        """
        Implement insM operator (Insertion Mutation).
        :return: one mutated chromosome
        """
        # Random gene and position for mutation
        permutation = np.random.permutation(len(self.chromosome))
        selected_gene_position, new_position = \
            permutation[0], permutation[1]
        # Identify the selected gene to be inserted into the new position
        selected_gene = self.chromosome[selected_gene_position]
        # Mutation operator logic
        self.mutated_chromosome = np.insert(
            arr=np.delete(
                arr=self.chromosome,
                obj=selected_gene_position
            ),
            obj=new_position,
            values=selected_gene
        )
        return self.mutated_chromosome

    def swap_m(self):
        """
        Implement swpM operator (Swap Mutation).
        :return: one mutated chromosome
        """
        # Random positions to participate in swapping
        permutation = np.random.permutation(len(self.chromosome))
        position_1, position_2 = \
            permutation[0], permutation[1]
        # Mutation operator logic
        self.mutated_chromosome[position_1], self.mutated_chromosome[position_2] = \
            self.mutated_chromosome[position_2], self.mutated_chromosome[position_1]
        return self.mutated_chromosome

    def inversion_m(self):
        """
        Implement invM operator (Inversion Mutation).
        :return: one mutated chromosome
        """
        # Random permutation on positions to avoid duplication in boundaries
        random_positions_permutation = np.random.permutation(len(self.chromosome))
        # Define inversion boundary
        low = random_positions_permutation[0]
        high = random_positions_permutation[1]
        if high < low:
            low, high = high, low
        # Mutation operator logic
        number_of_steps = int(np.ceil((high - low) / 2))
        for step in range(number_of_steps):
            self.mutated_chromosome[low + step], self.mutated_chromosome[high - step] = \
                self.mutated_chromosome[high - step], self.mutated_chromosome[low + step]
        return self.mutated_chromosome

    def adjacent_m(self):
        """
        Implement adjM operator (Adjacent jobs change Mutation).
        :return: one mutated chromosome
        """
        # Random high position
        high_position = np.random.randint(
            low=1,
            high=len(self.mutated_chromosome)
        )
        low = high_position - 1
        high = high_position
        # Mutation operator logic
        self.mutated_chromosome[low], self.mutated_chromosome[high] = \
            self.mutated_chromosome[high], self.mutated_chromosome[low]
        return self.mutated_chromosome

    def three_jobs_change_m(self):
        """
        Implement thjM operator (3-jobs change Mutation).
        :return: one mutated chromosome
        """
        # Select three random positions
        random_positions = np.random.permutation(len(self.mutated_chromosome))[:3]
        # Obtain a new permutation between selected positions
        permuted_random_positions = np.random.permutation(random_positions)
        # Mutation operator logic
        for position in range(len(random_positions)):
            self.mutated_chromosome[random_positions[position]], \
                self.mutated_chromosome[permuted_random_positions[position]] = \
                self.mutated_chromosome[permuted_random_positions[position]], \
                    self.mutated_chromosome[random_positions[position]]
        return self.mutated_chromosome


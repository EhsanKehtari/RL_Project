import numpy as np
from numpy import ndarray
import random


class Crossover:
    """
    Implement well-known crossovers in flowshop problem.
    List of crossovers:
    - NXO
    """

    def __init__(self, chromosome_1: ndarray, chromosome_2: ndarray, job_machine_matrix: None):
        """
        :param chromosome_1: one parent to be considered for reproducing offsprings
        :param chromosome_2: one parent to be considered for reproducing offsprings
        :param job_machine_matrix: matrix of process times of jobs on machines;
                                   shape: (number of jobs, number of machines)
        """
        self.chromosome_1 = chromosome_1
        self.chromosome_2 = chromosome_2
        # Pre-mature offsprings
        self.offspring_1 = np.zeros_like(self.chromosome_1)
        self.offspring_2 = np.zeros_like(self.chromosome_1)
        self.job_machine_matrix = job_machine_matrix

    def nxo(self):
        # NXO requirements checking point
        assert not (self.job_machine_matrix is None), "NXO crossover requires job machine matrix to be specified."
        # For parent1 selection at random
        parent_random_number = random.random()
        # Offsprings reproduction
        for i in range(2):
            if parent_random_number <= 0.5:
                parent_1, parent_2= self.chromosome_1, self.chromosome_2
                selected_offspring = self.offspring_1
            else:
                parent_2, parent_1 = self.chromosome_1, self.chromosome_2
                selected_offspring = self.offspring_2
            # Initialize the first gene of the selected_offspring
            selected_offspring[0] = parent_1[0]







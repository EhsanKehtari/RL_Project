import numpy as np
from numpy import ndarray
import random


class Crossover_helper:
    def __init__(self):
        return

    @staticmethod
    def fittest(gene_1: int, gene_2: int, job_machine_matrix: ndarray, jobs: ndarray) -> int:
        """
        Find the fittest gene among the two gens if both are available (NXO Crossover)
        :param gene_1: a gene to be considered for fitness
        :param gene_2: a gene to be considered for fitness
        :param job_machine_matrix: matrix of process times of jobs on machines;
                                   shape: (number of jobs, number of machines)
        :return: the fittest gene
        """
        # Find genes' corresponding rows in job_machine_matrix
        gene_1_location = int(np.argwhere(jobs == gene_1))
        gene_2_location = int(np.argwhere(jobs == gene_2))
        # Check fitness criteria
        # Pre-operative
        if job_machine_matrix[gene_1_location, 0] > job_machine_matrix[gene_2_location, 0]:
            return gene_1
        elif job_machine_matrix[gene_1_location, 0] < job_machine_matrix[gene_2_location, 0]:
            return gene_2
        else:
            # Peri-operative
            if job_machine_matrix[gene_1_location, 1] > job_machine_matrix[gene_2_location, 1]:
                return gene_1
            elif job_machine_matrix[gene_1_location, 1] < job_machine_matrix[gene_2_location, 1]:
                return gene_2
            else:
                #  Post-operative
                if job_machine_matrix[gene_1_location, 2] > job_machine_matrix[gene_2_location, 2]:
                    return gene_1
                elif job_machine_matrix[gene_1_location, 2] < job_machine_matrix[gene_2_location, 2]:
                    return gene_2
                else:
                    return gene_1

class Crossover:
    """
    Implement well-known crossovers in flowshop problem.
    List of crossovers:
    - NXO
    """

    def __init__(self, chromosome_1: ndarray, chromosome_2: ndarray, job_machine_matrix: None, jobs: None):
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
        self.jobs = jobs

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
            # Initialize and record the first gene of the selected_offspring
            selected_offspring[0] = parent_1[0]
            selected_gene = parent_1[0]
            # Put other genes in the selected_offspring
            while selected_offspring[-1] != 0:
                next_gene_1 =








from typing import Tuple

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

    @staticmethod
    def identify_next_gene(current_gene: int, chromosome: ndarray, offspring: ndarray) -> Tuple[bool, int]:
        """
        Identify availability of the next gene in a chromosome
        :param current_gene: current gene
        :param chromosome: the chromosome we are interested to study
        :param offspring: the offspring that affects the availability of the next gene
        :return: a tuple indicating whether the immediate gene following the current gene is available
                 and the next gene available

        return types
        1. (True, X): the immediate gene following the current gene is available and it is X.
        2. (False, Y): the immediate gene following the current gene is NOT available but the next available gene is Y.
        """
        # Locate the current gene in the given chromosome
        current_gene_location = int(np.argwhere(chromosome == current_gene))
        # availability criteria 1
        if current_gene_location + 1 > len(chromosome) - 1:
            _, next_available_gene = Crossover_helper().identify_next_gene(
                current_gene=int(chromosome[(current_gene_location + 1) % (len(chromosome) - 1)]),
                chromosome=chromosome,
                offspring=offspring
            )
            return False, next_available_gene
        elif chromosome[current_gene_location + 1] in offspring:
            _, next_available_gene = Crossover_helper().identify_next_gene(
                current_gene=int(chromosome[(current_gene_location + 1) % (len(chromosome) - 1)]),
                chromosome=chromosome,
                offspring=offspring
            )
            return False, next_available_gene
        else:
            next_available_gene = chromosome[(current_gene_location + 1)]
            return True, next_available_gene

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
            # Populate selected_offspring with other remaining genes
            while selected_offspring[-1] != 0:
                # Locate the selected gene in the parents
                selected_gene_location_in_parent_1 = int(np.argwhere(parent_1 == selected_gene))
                selected_gene_location_in_parent_2 = int(np.argwhere(parent_2 == selected_gene))
                next_gene_1 =








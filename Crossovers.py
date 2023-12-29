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
        :param jobs: jobs with associated numbers; shape: (number of jobs,)
        :return: the fittest gene
        """
        # Same genes have the same fitness
        if gene_1 == gene_2:
            return gene_1
        else:
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
    def identify_next_gene(current_gene: int,
                           chromosome: ndarray,
                           offspring: ndarray,
                           current_gene_checked=True) -> Tuple[bool, int]:
        """
        Identify availability of the next gene in a chromosome
        :param current_gene: current gene
        :param chromosome: the chromosome we are interested to study
        :param offspring: the offspring that affects the availability of the next gene
        :param current_gene_checked: to determine whether current gene has already been checked for availability
        :return: a tuple indicating whether the immediate gene following the current gene is available
                 and the next gene available

        return types
        1. (True, X): the immediate gene following the current gene is available and it is X.
        2. (False, Y): the immediate gene following the current gene is NOT available but the next available gene is Y.
        """
        # Locate the current gene in the given chromosome
        current_gene_location = int(np.argwhere(chromosome == current_gene))
        # Whether current gene has already been checked
        if current_gene_checked:
            # availability criteria
            if (current_gene_location + 1 > len(chromosome) - 1) or (
                    chromosome[current_gene_location + 1] in offspring):
                _, next_available_gene = Crossover_helper().identify_next_gene(
                    current_gene=int(chromosome[(current_gene_location + 1) % len(chromosome)]),
                    chromosome=chromosome,
                    offspring=offspring,
                    current_gene_checked=False
                )
                return False, next_available_gene
            else:
                next_available_gene = chromosome[(current_gene_location + 1)]
                return True, next_available_gene
        else:
            # availability criteria
            if chromosome[current_gene_location] in offspring:
                _, next_available_gene = Crossover_helper().identify_next_gene(
                    current_gene=int(chromosome[(current_gene_location + 1) % len(chromosome)]),
                    chromosome=chromosome,
                    offspring=offspring,
                    current_gene_checked=False
                )
                return False, next_available_gene
            else:
                next_available_gene = chromosome[current_gene_location]
                return True, next_available_gene

    @staticmethod
    def mapping_relationship(gene: int, mapping_section_1: ndarray, mapping_section_2: ndarray) -> int:
        """
        Determine mapping relationship for a PMX crossover.
        :param gene: the gene we are interested to find the mapping relationship for
        :param mapping_section_1: mapping section derived from chromosome 1 (gene's chromosome)
        :param mapping_section_2: mapping section derived from chromosome 2
        :return: related gene
        """
        temp_gene = gene
        while temp_gene in mapping_section_2:
            common_gene_arg = int(np.argwhere(mapping_section_2 == temp_gene))
            temp_gene = mapping_section_1[common_gene_arg]
        return temp_gene


class Crossover:
    """
    Implement well-known crossovers in flowshop problem.
    List of crossovers:
    - NXO
    """

    def __init__(self, chromosome_1: ndarray, chromosome_2: ndarray, job_machine_matrix=None, jobs=None):
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

    def nxo(self) -> Tuple[ndarray, ndarray]:
        """
        Implement NXO crossover.
        :return: (tuple) two offsprings reproduced from given parents.
        """
        # NXO requirements checking point
        assert not (self.job_machine_matrix is None), "NXO crossover requires job machine matrix to be specified."
        # For parent1 selection at random
        parent_random_number = random.random()
        # Offsprings reproduction
        for i in range(2):
            if parent_random_number <= 0.5:
                parent_1, parent_2 = self.chromosome_1, self.chromosome_2
                selected_offspring = self.offspring_1
            else:
                parent_2, parent_1 = self.chromosome_1, self.chromosome_2
                selected_offspring = self.offspring_2
            # next gene location in offspring
            next_gene_location_in_selected_offspring = 0
            # Initialize and record the first gene of the selected_offspring
            selected_gene = parent_1[next_gene_location_in_selected_offspring]
            selected_offspring[next_gene_location_in_selected_offspring] = selected_gene
            # Populate selected_offspring with other remaining genes
            while next_gene_location_in_selected_offspring <= len(selected_offspring) - 2:
                next_gene_location_in_selected_offspring += 1
                # Parent 1
                next_gene_1_available, next_gene_1 = Crossover_helper().identify_next_gene(
                    current_gene=selected_gene,
                    chromosome=parent_1,
                    offspring=selected_offspring
                )
                # Parent 2
                next_gene_2_available, next_gene_2 = Crossover_helper().identify_next_gene(
                    current_gene=selected_gene,
                    chromosome=parent_2,
                    offspring=selected_offspring
                )
                # Deciding on which gene to allocate
                if next_gene_1_available and next_gene_2_available:
                    temp = Crossover_helper().fittest(
                        gene_1=next_gene_1,
                        gene_2=next_gene_2,
                        job_machine_matrix=self.job_machine_matrix,
                        jobs=self.jobs
                    )
                elif next_gene_1_available and not next_gene_2_available:
                    temp = next_gene_1
                elif not next_gene_1_available and next_gene_2_available:
                    temp = next_gene_2
                else:
                    temp = Crossover_helper().fittest(
                        gene_1=next_gene_1,
                        gene_2=next_gene_2,
                        job_machine_matrix=self.job_machine_matrix,
                        jobs=self.jobs
                    )
                # Update selected_gene with temp as the next selected gene
                selected_gene = temp
                # Update selected_offspring with the new selected_gene
                selected_offspring[next_gene_location_in_selected_offspring] = selected_gene
            # To trigger the reproduction of the other offspring
            parent_random_number = 1 - parent_random_number
        return self.offspring_1, self.offspring_2

    def pmx(self) -> Tuple[ndarray, ndarray]:
        """
        Implement PMX crossover.
        :return: (tuple) two offsprings reproduced from given parents.
        """
        # Random positions to indicate the mapping section
        random_positions = np.random.randint(
            low=0,
            high=len(self.offspring_1),
            size=2
        )
        # Avoid duplicates
        while random_positions[0] == random_positions[1]:
            random_positions = np.random.randint(
                low=0,
                high=len(self.offspring_1),
                size=2
            )
        # Lower and upper bound for mapping section
        low = random_positions.min()
        high = random_positions.max()
        # Populating offsprings
        for gene_num in range(len(self.offspring_1)):
            # Mapping Section
            if low <= gene_num <= high:
                self.offspring_1[gene_num] = self.chromosome_2[gene_num]
                self.offspring_2[gene_num] = self.chromosome_1[gene_num]
            # For genes not in mapping section
            else:
                # Offspring 1
                if self.chromosome_1[gene_num] in self.chromosome_2[low: high + 1]:
                    self.offspring_1[gene_num] = Crossover_helper().mapping_relationship(
                        gene=self.chromosome_1[gene_num],
                        mapping_section_1=self.chromosome_1[low: high + 1],
                        mapping_section_2=self.chromosome_2[low: high + 1]
                    )
                else:
                    self.offspring_1[gene_num] = self.chromosome_1[gene_num]
                # Offspring 2
                if self.chromosome_2[gene_num] in self.chromosome_1[low: high + 1]:
                    self.offspring_2[gene_num] = Crossover_helper().mapping_relationship(
                        gene=self.chromosome_2[gene_num],
                        mapping_section_1=self.chromosome_2[low: high + 1],
                        mapping_section_2=self.chromosome_1[low: high + 1]
                    )
                else:
                    self.offspring_2[gene_num] = self.chromosome_2[gene_num]
        return self.offspring_1, self.offspring_2

    def pbx(self, number_of_positions: int) -> Tuple[ndarray, ndarray]:
        """
        Implement PBX crossover.
        :param number_of_positions: the number of positions to be selected at random.
        :return: (tuple) two offsprings reproduced from given parents.
        """
        # Sanity Check
        assert number_of_positions < len(self.offspring_1), 'Less number of positions expected.'
        # Random permutation on positions
        random_positions_permutation = list(
            np.random.permutation(len(self.offspring_1))
        )
        # Selecting the first number_of_positions as special positions
        special_positions = random_positions_permutation[:number_of_positions]
        # For parent1 selection at random
        parent_random_number = random.random()
        for i in range(2):
            if parent_random_number <= 0.5:
                parent_1, parent_2 = self.chromosome_1, self.chromosome_2
                selected_offspring = self.offspring_1
            else:
                parent_2, parent_1 = self.chromosome_1, self.chromosome_2
                selected_offspring = self.offspring_2
            # Two lists to store genes in special positions and genes other than those in special positions
            genes_in_special_positions = list()
            genes_in_non_special_positions = list()
            # Find genes in special positions
            for position in range(len(self.offspring_1)):
                if position in special_positions:
                    genes_in_special_positions.append(parent_1[position])
            # Find genes other that those in special positions
            for gene in parent_2:
                if not (gene in genes_in_special_positions):
                    genes_in_non_special_positions.append(gene)
            # Offspring reproduction logic
            for gene_num in range(len(selected_offspring)):
                if gene_num in special_positions:
                    selected_offspring[gene_num] = parent_1[gene_num]
                else:
                    selected_offspring[gene_num] = genes_in_non_special_positions.pop(0)
            # To trigger the reproduction of the other offspring
            parent_random_number = 1 - parent_random_number
        return self.offspring_1, self.offspring_2

    def ox(self) -> Tuple[ndarray, ndarray]:
        """
        Implement OX crossover.
        :return: (tuple) two offsprings reproduced from given parents.
        """
        # Random positions to indicate the mapping section
        random_positions = np.random.randint(
            low=0,
            high=len(self.offspring_1),
            size=2
        )
        # Avoid duplicates
        while random_positions[0] == random_positions[1]:
            random_positions = np.random.randint(
                low=0,
                high=len(self.offspring_1),
                size=2
            )
        # Lower and upper bound for substring selection
        low = random_positions.min()
        high = random_positions.max()
        print(low + 1, high + 1)
        # For parent1 selection at random
        parent_random_number = random.random()
        for i in range(2):
            if parent_random_number <= 0.5:
                parent_1, parent_2 = self.chromosome_1, self.chromosome_2
                selected_offspring = self.offspring_1
            else:
                parent_2, parent_1 = self.chromosome_1, self.chromosome_2
                selected_offspring = self.offspring_2
            # Two lists to store genes in special positions and genes other than those in special positions
            genes_in_special_positions = parent_1[low: high + 1]
            genes_in_non_special_positions = list()
            # Find genes other that those in special positions
            for gene in parent_2:
                if not (gene in genes_in_special_positions):
                    genes_in_non_special_positions.append(gene)
            # Offspring reproduction logic
            for gene_num in range(len(selected_offspring)):
                if gene_num in range(low, high + 1):
                    selected_offspring[gene_num] = parent_1[gene_num]
                else:
                    selected_offspring[gene_num] = genes_in_non_special_positions.pop(0)
            # To trigger the reproduction of the other offspring
            parent_random_number = 1 - parent_random_number
        return self.offspring_1, self.offspring_2


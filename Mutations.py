import numpy as np
from numpy import ndarray


class Mutation:
    """
    Implement well-known mutation operators in flowshop problem.
    List of mutation operators:
    - ins_m
    """

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.mutated_chromosome = self.chromosome

    def ins_m(self) -> ndarray:
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

    def swp_m(self):
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

    def inv_m(self):
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

    def adj_m(self):
        """
        Implement adjM operator (Adjacent two job change Mutation).
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

chro_1 = np.array(
    [1, 2, 3, 4, 5, 6, 7]
)
mut = Mutation(chromosome=chro_1)
print(mut.adj_m())

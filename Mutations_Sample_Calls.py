import numpy as np
from Mutations import Mutation


chromosome = np.array(
    [27, 18, 6, 52, 3, 1, 9]
)
mutation = Mutation(chromosome=chromosome)
# Insertion Mutation Operator
print('Insertion Mutation Operator: ', '\n', mutation.insertion_m())
# Swap Mutation Operator
print('Swap Mutation Operator: ', '\n', mutation.swap_m())
# Inversion Mutation Operator
print('Inversion Mutation Operator: ', '\n', mutation.inversion_m())
# Adjacent Mutation Operator
print('Adjacent Mutation Operator: ', '\n', mutation.adjacent_m())
# Three Jobs Change Mutation Operator
print('Three Jobs Change Mutation Operator: ', '\n', mutation.three_jobs_change_m())
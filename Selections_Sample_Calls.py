import numpy as np
from Selections import Selection

population = np.array(
    [[1, 13, 6, 9, 5],
     [13, 5, 1, 9, 6],
     [6, 5, 1, 13, 9],
     [1, 9, 6, 5, 13],
     [9, 5, 13, 1, 6],
     [13, 5, 9, 1, 6]]
)
obj_func_val = np.array(
    [11.5, 3, 14, 17, 1.5, 9]
)
rank_val = np.array(
    [6/21, 5/21, 4/21, 3/21, 2/21, 1/21]
)
selection_size = 6
selection = Selection(
    population=population,
    objective_function_values=obj_func_val,
    rank_values=rank_val,
    selection_size=selection_size
)
# Random Selection
print('Random Selection: ', '\n', selection.random_s())
# Greedy Selection
print('Greedy Selection: ', '\n', selection.greedy_s())
# Cost Weighted Roulette Wheel Selection
print('Cost Weighted Roulette Wheel Selection: ', '\n', selection.cost_weighted_roulette_wheel_s())
# Rank Weighted Roulette Wheel Selection
print('Rank Weighted Roulette Wheel Selection: ', '\n', selection.rank_weighted_roulette_wheel_s())
# Tournament Selection
print('Tournament Selection: ', '\n', selection.tournament_s())

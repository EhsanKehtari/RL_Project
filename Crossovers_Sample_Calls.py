import numpy as np
from Crossovers import Crossover

chromosome_1 = np.array(
    [5, 11, 21, 4, 9, 6, 16]
)
chromosome_2 = np.array(
    [21, 16, 5, 4, 11, 9, 6]
)
job_machine_matrix = np.array(
    [[1.00, 12.5, 3.50, 6.80],
     [5.00, 2.21, 9.88, 51.00],
     [6.20, 3.11, 18.20, 14.10],
     [23.23, 11.50, 6.74, 17.00],
     [8.25, 8.64, 11.25, 9.00],
     [5.00, 4.12, 3.01, 2.11],
     [9.11, 4.36, 8.54, 10.10]]
)
jobs = np.array(
    [5, 6, 4, 21, 9, 11, 16]
)
crossover = Crossover(
    chromosome_1=chromosome_1,
    chromosome_2=chromosome_2,
    job_machine_matrix=job_machine_matrix,
    jobs=jobs
)
# NXO
print('NXO Crossover: ', '\n', crossover.nxo())
# Partially Matched Crossover
print('Partially Mapped Crossover: ', '\n', crossover.pmx())
# Position Based Crossover
print('Position Based Crossover: ', '\n', crossover.pbx())
# Cycle Crossover
print('Cycle Crossover: ', '\n', crossover.cx())
# Order Based Crossover
print('Order Based Crossover: ', '\n', crossover.obx())
# One Point Crossover
print('One Point Crossover: ', '\n', crossover.opx())
# Modified One Point Crossover
print('Modified One Point Crossover: ', '\n', crossover.mopx())
# Two Point Crossover
print('Two Point Crossover: ', '\n', crossover.tpx())
# Similar Job Order Crossover
print('Similar Job Order Crossover: ', '\n', crossover.sjox())
# (Linear) Order Crossover
print('(Linear) Order Crossover: ', '\n', crossover.ox())
# Modified (Linear) Order Crossover
print('Modified (Linear) Order Crossover: ', '\n', crossover.mox())
# New Order Crossover
print('New Order Crossover: ', '\n', crossover.nox())
# Modified New Order Crossover
print('Modified New Order Crossover: ', '\n', crossover.mnox())

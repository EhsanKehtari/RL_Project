from typing import Tuple, Optional

import gym
from gym import spaces
import numpy as np
from gym.core import ObsType, ActType
from numpy import ndarray
from OTR_Simulation import Simulators
from Crossovers import Crossover
from Mutations import Mutation
from Selections import Selection
from Non_Static_Methods import get_non_static_methods_instances


class OTR_ENV2(gym.Env):
    """
    Operating Theater Room - Environment 2
    """
    def __init__(
            self,
            job_machine_matrix: ndarray,
            jobs: ndarray,
            stage_num_name_dict: dict,
            stage_machines_dict: dict,
            population_size: int,
            crossover_rate: float,
            mutation_rate: float,
            max_step_count: int
    ):
        """
        Initialize Environment 2.
        :param job_machine_matrix: matrix of process times of jobs on machines;
                                   shape: (number of jobs, number of machines).
        :param jobs: jobs' dedicated numbers (in order of appearance in job_machine_matrix).
        :param stage_num_name_dict: a dict with stage numbers as keys and stage names as values.
        :param stage_machines_dict: a dict with stage numbers as keys and number of machines at each stage as values.
        :param population_size: size of the initial population in genetic algorithm.
        :param crossover_rate: probability of applying crossover operators on individuals in a population.
        :param mutation_rate: probability of applying mutation operators on individuals in a population.
        :param max_step_count: maximum number of steps.
        """
        # Necessary attributes
        self.job_machine_matrix = job_machine_matrix
        self.jobs = jobs
        self.stage_num_name_dict = stage_num_name_dict
        self.stage_machines_dict = stage_machines_dict
        self.population_size = population_size
        self.number_of_jobs = len(self.jobs)
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_step_count = max_step_count
        # Observation space
        self.observation_space = spaces.Box()
        # Action space (Crossover / Mutation)
        self.action_space = spaces.MultiDiscrete(
            [len(get_non_static_methods_instances(Crossover())),
             len(get_non_static_methods_instances(Mutation()))]
        )
        # To check if job_machine_matrix is compatible with jobs
        assert self.job_machine_matrix.shape[0] == self.number_of_jobs, \
            'Unmatched number of jobs!'
        # Crossover rate interpretation
        if np.floor(self.crossover_rate * self.population_size) % 2 == 0:
            self.crossover_rate_interpreted = int(np.floor(self.crossover_rate * self.population_size))
        else:
            self.crossover_rate_interpreted = int(np.ceil(self.crossover_rate * self.population_size))
        # Mutation rate interpretation
        self.mutation_rate_interpreted = max(1, round(self.mutation_rate * self.population_size))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,) -> Tuple[ObsType, dict]:
        """
        Reset the environment to its initial state.
        """
        # Initial population
        self.population = np.zeros(shape=(self.population_size, self.number_of_jobs), dtype=int)
        self.c_max = np.zeros(shape=self.number_of_jobs)
        # Populate self.population
        for individual in range(self.population_size):
            self.population[individual] = np.random.permutation(self.jobs)
            # Initialize Simulators class
            simulators = Simulators(
                job_machine_matrix=self.job_machine_matrix,
                jobs=self.jobs,
                stage_num_name_dict=self.stage_num_name_dict,
                stage_machines_dict=self.stage_machines_dict,
                jobs_sequence=self.population[individual]
            )
            # LS simulator
            self.c_max[individual] = simulators.ls_simulator()
        # Step counter for problem termination
        self.step_count = 0
        self.problem_terminated = False
        # Observation space
        observation = None
        return observation, None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        Evolve one step further in the environment.
        """
        # Step counter for problem termination
        self.step_count += 1
        # Initialize selection procedure
        selection = Selection(
            population=self.population,
            objective_function_values=self.c_max,
            selection_size=self.population_size
        )
        # Create mating pool
        mating_pool = selection.cost_weighted_roulette_wheel_s()
        # Offspring empty population
        crossover_based_offspring_population = np.zeros(
            shape=(self.crossover_rate_interpreted, self.number_of_jobs),
            dtype=int
        )
        crossover_based_offspring_population_cmax = np.zeros(
            shape=self.crossover_rate_interpreted,
            dtype=float
        )
        mutation_based_offspring_population = np.zeros(
            shape=(self.mutation_rate_interpreted, self.number_of_jobs),
            dtype=int
        )
        mutation_based_offspring_population_cmax = np.zeros(
            shape=self.mutation_rate_interpreted,
            dtype=float
        )
        # Populate crossover-based offspring population;
        # Get random permutation on individuals in population for crossover implementation
        mating_pool_permuted = np.random.permutation(mating_pool)
        for couple in range(int(self.crossover_rate_interpreted / 2)):
            # Define parents
            father = mating_pool_permuted[2 * couple]
            mother = mating_pool_permuted[2 * couple + 1]
            # Execute crossover as the first element of step's action
            boy, girl = get_non_static_methods_instances(
                Crossover(
                    chromosome_1=father,
                    chromosome_2=mother,
                    job_machine_matrix=self.job_machine_matrix,
                    jobs=self.jobs
                )
            )[action[0]]()
            # Compute c_max based on LS Algorithm
            boy_cmax = Simulators(
                job_machine_matrix=self.job_machine_matrix,
                jobs=self.jobs,
                stage_num_name_dict=self.stage_num_name_dict,
                stage_machines_dict=self.stage_machines_dict,
                jobs_sequence=boy
            ).ls_simulator()
            girl_cmax = Simulators(
                job_machine_matrix=self.job_machine_matrix,
                jobs=self.jobs,
                stage_num_name_dict=self.stage_num_name_dict,
                stage_machines_dict=self.stage_machines_dict,
                jobs_sequence=girl
            ).ls_simulator()
            # Update crossover-based offspring population and corresponding c_max values
            crossover_based_offspring_population[2 * couple],\
                crossover_based_offspring_population[2 * couple + 1] = \
                boy,\
                    girl
            crossover_based_offspring_population_cmax[2 * couple],\
                crossover_based_offspring_population_cmax[2 * couple + 1] = \
                boy_cmax,\
                    girl_cmax
        # Populate mutation-based offspring population;
        # Get random permutation on individuals in population for mutation implementation
        mating_pool_permuted = np.random.permutation(mating_pool)
        for mutate_num in range(self.mutation_rate_interpreted):
            # Execute mutation as the second element of step's action
            mutation_based_offspring_population[mutate_num] = get_non_static_methods_instances(
                Mutation(
                    chromosome=mating_pool_permuted[mutate_num]
                )
            )[action[1]]()
            # Compute c_max based on LS Algorithm
            mutation_based_offspring_population_cmax[mutate_num] = Simulators(
                job_machine_matrix=self.job_machine_matrix,
                jobs=self.jobs,
                stage_num_name_dict=self.stage_num_name_dict,
                stage_machines_dict=self.stage_machines_dict,
                jobs_sequence=mutation_based_offspring_population[mutate_num]
            ).ls_simulator()
        # Expand population and c_max values by new offsprings (row-wise)
        population_expanded = np.concatenate(
            (self.population, crossover_based_offspring_population, mutation_based_offspring_population),
            axis=0
        )
        c_max_expanded = np.concatenate(
            (self.c_max, crossover_based_offspring_population_cmax, mutation_based_offspring_population_cmax),
            axis=0
        )
        self.population = Selection(
            population=population_expanded,
            objective_function_values=c_max_expanded,
            selection_size=self.population_size
        ).greedy_s()
        # Observation space
        observation = None
        # Reward function
        reward = None
        # Check problem termination condition
        if self.step_count == self.max_step_count:
            self.problem_terminated = True
        return observation, reward, self.problem_terminated, False, None


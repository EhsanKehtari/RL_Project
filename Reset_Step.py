import gym
from gym import spaces
import numpy as np


class OperatingRoomScheduling(gym.Env):
    def __init__(self, job_machine_matrix, stages_machines: list):
        self.observation_space = spaces.Box(low=0, high=np.infty, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(12)
        self.job_machine_matrix = job_machine_matrix
        self.number_of_stages = self.job_machine_matrix.shape[1]
        self.number_of_patients = self.job_machine_matrix.shape[0]
        self.stages_machines = stages_machines

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def end_of_pre_operative(self):
        pass
    
    def end_of_peri_operative(self):
        pass

    def end_of_post_operative(self):
        pass

    def reset(self):
        patients_list = list()
        for patients in range(self.number_of_patients):
            patient = Patient(patients + 1,
                              self.job_machine_matrix[patients, 0],
                              self.job_machine_matrix[patients, 1],
                              self.job_machine_matrix[patients, 2])
            patients_list.append(patient)
        machines_dict = dict()
        for stages in range(self.number_of_stages):
            machines_dict[stages] = list()
            for resources in range(self.stages_machines[stages]):
                resource = Resource(resources)
                machines_dict[stages].append(resource)

    def step(self, action):
        pass


class Patient:
    """

    """

    def __init__(self, identification, pre_operating_time, peri_operating_time, post_operating_time):
        self.id = identification
        self.block = False
        self.service_condition = np.array([0, 0, 0])
        self.pre_operating_time = pre_operating_time
        self.peri_operating_time = peri_operating_time
        self.post_operating_time = post_operating_time
        self.start_pre_operating = 0
        self.start_peri_operating = 0
        self.start_post_operating = 0
        self.end_pre_operating = self.start_pre_operating + self.pre_operating_time
        self.end_peri_operating = self.start_peri_operating + self.peri_operating_time
        self.end_post_operating = self.start_post_operating + self.post_operating_time

    def tmp(self):
        pass


class Resource:
    """

    """

    def __init__(self, identification, rate=1):
        self.id = identification
        self.status = 0
        self.block = False
        self.rate = rate
        self.sequence = list()

    def tmp(self):
        pass




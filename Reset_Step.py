from typing import Tuple

import gym
from gym import spaces
import numpy as np
from numpy import ndarray


class OperatingRoomScheduling(gym.Env):
    def __init__(self, job_machine_matrix: ndarray, jobs: ndarray, stages_machines: list):
        self.observation_space = spaces.Box(low=0, high=np.infty, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(12)
        self.job_machine_matrix = job_machine_matrix
        self.jobs = jobs
        self.number_of_stages = self.job_machine_matrix.shape[1]
        self.number_of_patients = self.job_machine_matrix.shape[0]
        # A dict to contain number of machines in each stage
        self.stages_machines = dict()
        for stage in range(len(stages_machines)):
            self.stages_machines['Stage ' + str(stage + 1)] = stages_machines[stage]
        # A dict to know a list of idle resources and waiting patients for each stage
        # to check whether necessary to take action
        # key: stage, value: dict --> key: idle resources, waiting patients  value: idle resources (list), waiting patients (list)
        self.take_action_info = dict()
        # A dict with keys corresponding to stages and values (as type nparray) corresponding to
        # waiting patients behind each stage
        self.waiting_patients_behind_stages = dict()
        self.machines_dict = dict()
        # A list to store instances of class Patient
        self.patients_list = list()
        self.future_event_list = list()
        self.clock = 0

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

    def reset(self):
        self.future_event_list = list()
        self.clock = 0
        # Initialize take_action_info
        for stage in range(self.number_of_stages):
            self.take_action_info['Stage ' + str(stage + 1)] = dict()
            self.take_action_info['Stage ' + str(stage + 1)]['Idle Resources'] = list()
            self.take_action_info['Stage ' + str(stage + 1)]['Waiting Patients'] = list()
            # In the beginning, all resources in all stages are idle
            self.take_action_info['Stage ' + str(stage + 1)]['Idle Resources'].extend(
                [resource for resource in range(1, self.stages_machines['Stage ' + str(stage + 1)])]
            )
            # In the beginning, only stage 1 has waiting patients
            if stage == 0:
                self.take_action_info['Stage ' + str(stage + 1)]['Waiting Patients'].extend(self.jobs.tolist())

        # Instantiating patients
        for patients in range(self.number_of_patients):
            patient = Patient(
                self.jobs[patients],
                self.job_machine_matrix[patients, 0],
                self.job_machine_matrix[patients, 1],
                self.job_machine_matrix[patients, 2]
            )
            self.patients_list.append(patient)

        # Instantiating resources
        for stages in range(self.number_of_stages):
            # In the beginning, all patients are waiting to enter pre-operative stage (i.e., stage 1)
            if stages == 0:
                self.waiting_patients_behind_stages['Stage 1'] = self.jobs
            else:
                self.waiting_patients_behind_stages['Stage ' + str(stages + 1)] = None
            self.machines_dict['Stage ' + str(stages + 1)] = list()
            for resources in range(self.stages_machines[stages]):
                resource = Resource(identification=(stages + 1, resources + 1))
                self.machines_dict['Stage ' + str(stages + 1)].append(resource)

    def fel_maker(self, patient_id, event_type):
        if event_type == 'End of Pre-Operative':
            column_indicator = 0
        elif event_type == 'End of Peri-Operative':
            column_indicator = 1
        else:
            column_indicator = 2
        event_time = self.clock + self.job_machine_matrix[patient_id - 1, column_indicator]
        self.future_event_list.append({
            'Event Type': event_type,
            'Event Time': event_time,
            'Patient ID': patient_id
        })

    def update_take_action_info(self):
        for stage in list(self.machines_dict.keys()):
            # Find idle resources in each stage
            for resource in range(len(self.machines_dict[stage])):
                if not self.machines_dict[stage][resource].working_status:
                    self.take_action_info[stage]['Idle Resources'].append(self.machines_dict[stage][resource].id[1])
            # Find waiting patients behind each stage
            self.take_action_info[stage]['Waiting Patients'].extend(self.waiting_patients_behind_stages[stage])

    def end_of_pre_operative(self):
        pass

    def end_of_peri_operative(self):
        pass

    def end_of_post_operative(self):
        pass

    def action_to_heuristics(self, action):


    def step(self, action):
        self.update_take_action_info()
        while True:
            for stage in list(self.take_action_info.keys()):
                if len(self.take_action_info[stage]['Idle Resources']) != 0 and \
                        len(self.take_action_info[stage]['Waiting Patients']) != 0:
                    break
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

    def __init__(self, identification: Tuple, rate=1):
        # resource id is a tuple: (stage number, resource number)
        self.id = identification
        self.working_status = False
        self.block = False
        self.rate = rate
        self.job_under_process = None
        self.done_jobs_sequence = list()

    def tmp(self):
        pass

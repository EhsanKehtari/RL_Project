from typing import Tuple

import gym
from gym import spaces
import numpy as np
from numpy import ndarray
from Heuristics_OOP import Flowshop


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
        self.machines_dict = dict()
        # A list to store instances of class Patient
        self.patients_list = list()
        # A dict to store additional information
        self.info = dict()
        self.future_event_list = list()
        self.clock = 0

    def reset(self):
        self.future_event_list = list()
        self.clock = 0
        # Instantiate patients
        for patients in range(self.number_of_patients):
            patient = Patient(
                self.jobs[patients],
                self.job_machine_matrix[patients, 0],
                self.job_machine_matrix[patients, 1],
                self.job_machine_matrix[patients, 2]
            )
            self.patients_list.append(patient)

        # Instantiate resources
        for stages in range(self.number_of_stages):
            self.machines_dict['Stage ' + str(stages + 1)] = list()
            for resources in range(self.stages_machines['Stage ' + str(stages + 1)]):
                resource = Resource(identification=(stages + 1, resources + 1))
                self.machines_dict['Stage ' + str(stages + 1)].append(resource)

        # Initialize take_action_info
        for stage in range(self.number_of_stages):
            self.take_action_info['Stage ' + str(stage + 1)] = dict()
            self.take_action_info['Stage ' + str(stage + 1)]['Idle Resources'] = list()
            self.take_action_info['Stage ' + str(stage + 1)]['Waiting Patients'] = list()
            # In the beginning, all resources in all stages are idle
            self.take_action_info['Stage ' + str(stage + 1)]['Idle Resources'].extend(
                self.machines_dict['Stage ' + str(stage + 1)]
            )
            # In the beginning, all patients are waiting to enter pre-operative stage (i.e., stage 1)
            if stage == 0:
                self.take_action_info['Stage ' + str(stage + 1)]['Waiting Patients'].extend(
                    self.patients_list
                )

    def fel_maker(self, patient, resource, event_type):
        if event_type == 'End of Pre-Operative':
            # Resource's service rate changes pre-defined pre-operating time
            patient.pre_operating_time *= resource.rate
            event_time = patient.end_pre_operating
        elif event_type == 'End of Peri-Operative':
            # Resource's service rate changes pre-defined peri-operating time
            patient.peri_operating_time *= resource.rate
            event_time = patient.end_peri_operating
        else:
            # Resource's service rate changes pre-defined post-operating time
            patient.post_operating_time *= resource.rate
            event_time = patient.end_post_operating
        self.future_event_list.append({
            'Event Type': event_type,
            'Event Time': event_time,
            'Patient': patient,
            'Resource': resource
        })

    def end_of_pre_operative(self, patient, resource):
        # Modify patient's attributes
        patient.block = True
        patient.service_condition[0] = 1
        # Move patient to next stage's waiting patients list
        self.take_action_info['Stage 2']['Waiting Patients'].append(patient)

        # Modify resource's attribute
        resource.working_status = False
        resource.block = True
        resource.job_under_process = None
        resource.done_jobs_sequence.append(patient)
        # Move resource to idle resources list
        self.take_action_info['Stage 1']['Idle Resources'].append(resource)

    def end_of_peri_operative(self, patient, resource):
        # Modify patient's attributes
        patient.block = True
        patient.service_condition[1] = 1
        # Move patient to next stage's waiting patients list
        self.take_action_info['Stage 3']['Waiting Patients'].append(patient)

        # Modify resource's attribute
        resource.working_status = False
        resource.block = True
        resource.job_under_process = None
        resource.done_jobs_sequence.append(patient)
        # Move resource to idle resources list
        self.take_action_info['Stage 2']['Idle Resources'].append(resource)

    def end_of_post_operative(self, patient, resource):
        # Modify patient's attributes
        patient.block = False
        patient.service_condition[2] = 1

        # Modify resource's attribute
        resource.working_status = False
        resource.block = False
        resource.job_under_process = None
        resource.done_jobs_sequence.append(patient)
        # Move resource to idle resources list
        self.take_action_info['Stage 3']['Idle Resources'].append(resource)

    def action_to_heuristics(self, action):
        pass

    def step(self, action):
        problem_terminated = False
        step_terminated = False
        while not step_terminated:
            for stage in list(self.take_action_info.keys()):
                # Step termination condition
                if len(self.take_action_info[stage]['Idle Resources']) != 0 and \
                        len(self.take_action_info[stage]['Waiting Patients']) != 0:
                    self.info['Next Step Stage'] = stage
                    step_terminated = True
                    break
            if step_terminated:
                break
            elif len(self.future_event_list) == 0:
                problem_terminated = True
                break
            else:
                # Sort fel based on event times
                sorted_fel = sorted(self.future_event_list, key=lambda x: x['Event Time'])
                # Find imminent event
                current_event = sorted_fel[0]
                # Restore current event's info from current_event dict
                self.clock = current_event['Event Time']
                current_event_type = current_event['Event Type']
                current_event_patient = current_event['Patient']
                current_event_resource = current_event['Resource']
                # Execute events
                if current_event_type == 'End of Pre-Operative':
                    self.end_of_pre_operative(
                        current_event_patient,
                        current_event_resource
                    )
                elif current_event_type == 'End of Peri-Operative':
                    self.end_of_peri_operative(
                        current_event_patient,
                        current_event_resource
                    )
                elif current_event_type == 'End of Post-Operative':
                    self.end_of_post_operative(
                        current_event_patient,
                        current_event_resource
                    )
                # Remove current event from fel
                self.future_event_list.remove(current_event)

        return observation, reward, problem_terminated, False, None


class Patient:
    """

    """

    def __init__(self, identification, pre_operating_time, peri_operating_time, post_operating_time):
        self.id = identification
        self.block = False
        self.service_condition = np.array([0, 0, 0])
        self.dismissed = False
        if self.service_condition[2] == 1:
            self.dismissed = True
        self.current_stage = 0
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

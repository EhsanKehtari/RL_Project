from typing import Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy import ndarray
# from Heuristics_OOP import Flowshop


class OperatingRoomScheduling(gym.Env):
    def __init__(self, job_machine_matrix: ndarray, jobs: ndarray, stages_machines: list):
        self.observation_space = spaces.Box(low=0, high=np.infty, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(11)
        self.job_machine_matrix = job_machine_matrix
        self.jobs = jobs
        self.number_of_stages = self.job_machine_matrix.shape[1]
        # To check if job_machine_matrix is compatible with jobs
        assert self.job_machine_matrix.shape[0] == len(self.jobs), \
            'Unmatched number of jobs!'
        self.number_of_patients = self.job_machine_matrix.shape[0]
        # A dict to contain number of machines in each stage
        self.stages_machines = dict()
        for stage in range(len(stages_machines)):
            self.stages_machines['Stage ' + str(stage + 1)] = stages_machines[stage]
        # A dict to know a list of idle resources and waiting patients for each stage
        # to check whether necessary to take action
        # key: stage, value: dict -->
        # keys: idle resources, waiting patients  values: idle resources (list), waiting patients (list)
        self.take_action_info = dict()
        self.machines_dict = dict()
        # A list to store instances of class Patient
        self.patients_list = list()
        # A dict to store additional information
        self.info = dict()
        self.future_event_list = list()
        self.clock = 0

        self.episode_num = 0
        self.min_makespan = min_makespan
        self.previous_utilization = 0  # Initialize previous utilization


    def reset(self, seed = None, options = None):

        self.clock = 0
        self.previous_utilization = 0
        self.current_utilization = 0
        self.total_wait_time = 0
        self.total_reward = 0
        self.episode_num += 1

        self.future_event_list.clear()
        self.patients_list.clear()
        self.machines_dict.clear()
        self.take_action_info.clear()

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
        # Stage 1 is the next stage to be taken care of in the next step (i.e., step 1)
        self.info['Next Step Stage'] = 'Stage 1'

        observation = self._get_observation()
        return observation, {}

    def end_of_pre_operative(self, patient, resource):
        # Modify job's attributes
        patient.block = True
        patient.service_condition[0] = 1
        # Move job to next stage's waiting patients list
        self.take_action_info['Stage 2']['Waiting Patients'].append(patient)
        # Modify resource's attribute
        resource.working_status = False
        resource.block = True
        resource.job_under_process = None
        resource.done_jobs_sequence_pre.append(patient)
        # print(f"Added patient {patient.id} to done_jobs_sequence_pre")


    def end_of_peri_operative(self, patient, resource):
        # Modify job's attributes
        patient.block = True
        patient.service_condition[1] = 1
        # Move job to next stage's waiting patients list
        self.take_action_info['Stage 3']['Waiting Patients'].append(patient)
        # Modify resource's attribute
        resource.working_status = False
        resource.block = True
        resource.job_under_process = None
        resource.done_jobs_sequence_peri.append(patient)
        # print(f"Added patient {patient.id} to done_jobs_sequence_peri")


    def end_of_post_operative(self, patient, resource):
        # Modify job's attributes
        patient.block = False
        patient.service_condition[2] = 1
        # Modify resource's attribute
        resource.working_status = False
        resource.block = False
        resource.job_under_process = None
        resource.done_jobs_sequence_post.append(patient)
        # print(f"Added patient {patient.id} to done_jobs_sequence_post")

        # Move resource to idle resources list
        self.take_action_info['Stage 3']['Idle Resources'].append(resource)


    def action_to_heuristics(self, action):
        # Turn stage into number
        stage_under_study = self.info['Next Step Stage']
        if stage_under_study == 'Stage 1':
            numeric_stage_under_study = 0
        elif stage_under_study == 'Stage 2':
            numeric_stage_under_study = 1
        else:
            numeric_stage_under_study = 2

        # Get waiting patients
        waiting_patients = self.take_action_info[stage_under_study]['Waiting Patients']

        # Get waiting patients indices
        waiting_patients_ids = [patient.id for patient in waiting_patients]

        # No need to execute a heuristic if only one job is ready to undergo a process in any given stage
        if len(waiting_patients_ids) == 1:
            patient_idx = waiting_patients_ids[0]
        else:
            # Map actions to heuristics and execute the chosen heuristic
            if action == 0:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                cds_job_machine_matrix = self.job_machine_matrix[job_idx][:, numeric_stage_under_study:]
                cds_instance = Flowshop(cds_job_machine_matrix, np.array(waiting_patients_ids))
                cds_result_sequence = cds_instance.cds()
                patient_idx = cds_result_sequence[0]
            elif action == 1:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                neh_job_machine_matrix = self.job_machine_matrix[job_idx][:, numeric_stage_under_study:]
                neh_instance = Flowshop(neh_job_machine_matrix, np.array(waiting_patients_ids))
                neh_result_sequence = neh_instance.cds()
                patient_idx = neh_result_sequence[0]
            elif action == 2:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                mst_job_machine_matrix = self.job_machine_matrix[job_idx][:, 1]
                mst_instance = Flowshop(mst_job_machine_matrix, np.array(waiting_patients_ids))
                mst_result_sequence = mst_instance.mst()
                patient_idx = mst_result_sequence[0]
            elif action == 3:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                mot_job_machine_matrix = np.sum(self.job_machine_matrix[job_idx], axis=1)
                mot_instance = Flowshop(mot_job_machine_matrix, np.array(waiting_patients_ids))
                mot_result_sequence = mot_instance.mot()
                patient_idx = mot_result_sequence[0]
            elif action == 4:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                sespt_lespt_job_machine_matrix = self.job_machine_matrix[job_idx][:, 1]
                sespt_lespt_instance = Flowshop(sespt_lespt_job_machine_matrix, np.array(waiting_patients_ids))
                sespt_lespt_result_sequence = sespt_lespt_instance.sespt_lespt()
                patient_idx = sespt_lespt_result_sequence[0]
            elif action == 5:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                seopt_leopt_job_machine_matrix = np.sum(self.job_machine_matrix[job_idx], axis=1)
                seopt_leopt_instance = Flowshop(seopt_leopt_job_machine_matrix, np.array(waiting_patients_ids))
                seopt_leopt_result_sequence = seopt_leopt_instance.seopt_leopt()
                patient_idx = seopt_leopt_result_sequence[0]
            elif action == 6:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                lespt_sespt_job_machine_matrix = self.job_machine_matrix[job_idx][:, 1]
                lespt_sespt_instance = Flowshop(lespt_sespt_job_machine_matrix, np.array(waiting_patients_ids))
                lespt_sespt_result_sequence = lespt_sespt_instance.lespt_sespt()
                patient_idx = lespt_sespt_result_sequence[0]
            elif action == 7:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                leopt_seopt_job_machine_matrix = np.sum(self.job_machine_matrix[job_idx], axis=1)
                leopt_seopt_instance = Flowshop(leopt_seopt_job_machine_matrix, np.array(waiting_patients_ids))
                leopt_seopt_result_sequence = leopt_seopt_instance.leopt_seopt()
                patient_idx = leopt_seopt_result_sequence[0]
            elif action == 8:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                sespt_job_machine_matrix = self.job_machine_matrix[job_idx][:, 1]
                sespt_instance = Flowshop(sespt_job_machine_matrix, np.array(waiting_patients_ids))
                sespt_result_sequence = sespt_instance.sespt()
                patient_idx = sespt_result_sequence[0]
            elif action == 9:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                seopt_job_machine_matrix = np.sum(self.job_machine_matrix[job_idx], axis=1)
                seopt_instance = Flowshop(seopt_job_machine_matrix, np.array(waiting_patients_ids))
                seopt_result_sequence = seopt_instance.seopt()
                patient_idx = seopt_result_sequence[0]
            elif action == 10:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                lespt_job_machine_matrix = self.job_machine_matrix[job_idx][:, 1]
                lespt_instance = Flowshop(lespt_job_machine_matrix, np.array(waiting_patients_ids))
                lespt_result_sequence = lespt_instance.lespt()
                patient_idx = lespt_result_sequence[0]
            elif action == 11:
                job_idx = [np.argwhere(self.jobs == patient_id).flatten()[0] for patient_id in waiting_patients_ids]
                leopt_job_machine_matrix = np.sum(self.job_machine_matrix[job_idx], axis=1)
                leopt_instance = Flowshop(leopt_job_machine_matrix, np.array(waiting_patients_ids))
                leopt_result_sequence = leopt_instance.leopt()
                patient_idx = leopt_result_sequence[0]

        # Find chosen job object according to patient_idx
        chosen_patient_location = waiting_patients_ids.index(patient_idx)
        patient = waiting_patients[chosen_patient_location]
        # Specify an idle resource to take care of the chosen job
        resource = self.take_action_info[stage_under_study]['Idle Resources'][0]
        return patient, resource



    def schedule_patient(self, patient, resource, current_clock):
        # Modify job's attributes
        patient.block = False
        patient.current_stage = resource.id[0]
        if patient.current_stage == 1:
            # Patient starts getting service
            patient.start_pre_operating = current_clock
            # Resource's service rate changes pre-defined pre-operating time
            patient.pre_operating_time *= resource.rate
            # Patient's end of service changes according to clock and assigned resource's service rate
            patient.end_pre_operating = patient.start_pre_operating + patient.pre_operating_time
        elif patient.current_stage == 2:
            # Patient starts getting service
            patient.start_peri_operating = current_clock
            # Resource's service rate changes pre-defined pre-operating time
            patient.peri_operating_time *= resource.rate
            # Patient's end of service changes according to clock and assigned resource's service rate
            # Get the raw end time
            patient.end_peri_operating = patient.start_peri_operating + patient.peri_operating_time

            # if 0 < raw_end_time < 540:
            #     # If within the first 540 units, calculate end_peri_operating normally
            #     patient.end_peri_operating = raw_end_time
            # else:
            #     # For cases where raw_end_time is greater than 540
            #     for i in range(1, 100):  # Loop from 1 to 49, representing ranges above 540
            #         if i * 540 < raw_end_time < (i + 1) * 540:
            #             current_clock += (i * 1440)
            #             print(current_clock)
            #             patient.start_pre_operating = current_clock
            #             patient.end_peri_operating = patient.start_peri_operating + patient.peri_operating_time
            #             break


        else:
            # Patient starts getting service
            patient.start_post_operating = current_clock
            # Resource's service rate changes pre-defined pre-operating time
            patient.post_operating_time *= resource.rate
            # Patient's end of service changes according to clock and assigned resource's service rate
            patient.end_post_operating = patient.start_post_operating + patient.post_operating_time

        # Update resource processing time
        resource.total_processing_time += patient.end_post_operating - patient.start_pre_operating if patient.current_stage == 3 else \
        patient.end_peri_operating - patient.start_peri_operating if patient.current_stage == 2 else \
        patient.end_pre_operating - patient.start_pre_operating

        # Remove job from current stage's waiting patients list
        self.take_action_info['Stage ' + str(patient.current_stage)]['Waiting Patients'].remove(patient)
        # Modify resource's attributes
        resource.working_status = True
        resource.block = False
        resource.job_under_process = patient
        # Add previous resource to previous stage's idle resources list
        if 1 < resource.id[0]:
            self.take_action_info['Stage ' + str(resource.id[0] - 1)]['Idle Resources'].append(patient.assigned_resource)
        # Remove current resource from current stage's idle resources list
        self.take_action_info['Stage ' + str(resource.id[0])]['Idle Resources'].remove(resource)
        # Modify job's current assigned resource
        patient.assigned_resource = resource
        # Determine event type based on stage number
        if resource.id[0] == 1:
            event_type = 'End of Pre-Operative'
        elif resource.id[0] == 2:
            event_type = 'End of Peri-Operative'
        else:
            event_type = 'End of Post-Operative'
        # Schedule job on resource
        self.fel_maker(
            patient,
            resource,
            event_type
        )
        return current_clock

    def fel_maker(self, patient, resource, event_type):
        if event_type == 'End of Pre-Operative':
            event_time = patient.end_pre_operating
        elif event_type == 'End of Peri-Operative':
            event_time = patient.end_peri_operating
        else:
            event_time = patient.end_post_operating
        self.future_event_list.append({
            'Event Type': event_type,
            'Event Time': event_time,
            'Patient': patient,
            'Resource': resource
        })

    def step(self, action):
        # Interpret action and find a job to be scheduled on a resource
        patient, resource = self.action_to_heuristics(action)
        # Schedule job on the specified resource
        self.schedule_patient(patient, resource, self.clock)
        # Specify initial termination terms for problem and step
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
                episode_reward = self._calculate_episode_reward()

                # print(f"Episode {self.episode_num}, Makespan: {self.clock}, MinMakespan: {self.min_makespan}")

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

        observation = self._get_observation()
        step_reward = self._calculate_step_reward()
        reward = step_reward
        if problem_terminated:
            reward += episode_reward
        return observation, reward, problem_terminated, False, {}


    def _get_observation(self):
        # Current clock time
        current_time = self.clock

        # Number of future events
        num_future_events = len(self.future_event_list)

        # Number of waiting patients in each stage
        waiting_patients = [len(self.take_action_info[f'Stage {i+1}']['Waiting Patients'])
                            for i in range(self.number_of_stages)]

        # Number of idle resources in each stage
        idle_resources = [len(self.take_action_info[f'Stage {i+1}']['Idle Resources'])
                          for i in range(self.number_of_stages)]

        # Combine all observation components into a single array
        observation = np.array([
            current_time,
            num_future_events,
            *waiting_patients,
            *idle_resources
        ], dtype=np.float32)

        return observation


    def _calculate_episode_reward(self):
        # Compare current episode makespan with the minimum makespan= 0
        if self.clock <= self.min_makespan:
            episode_reward = 100
        else:
            episode_reward = 100/(self.clock-self.min_makespan)  # Penalty for not improving the makespan
        return episode_reward  # Reward for achieving a new minimum makespan


    def calculate_utilization(self):
        # for stage, stage_resources in self.machines_dict.items():
            # for resource in stage_resources:
                # print(f"Stage: {stage}, Resource: {resource}, Processing Time: {resource.total_processing_time}")
        total_processing_time = sum(
            resource.total_processing_time for stage_resources in self.machines_dict.values() for resource in stage_resources
        )
        if self.clock > 0:
            utilization = total_processing_time / self.clock
        else:
            utilization = 0
        return utilization

    def _calculate_step_reward(self):
        current_utilization = self.calculate_utilization()
        # print(current_utilization,self.previous_utilization)
        step_reward = (current_utilization - self.previous_utilization)
        self.previous_utilization = current_utilization  # Update previous utilization
        return step_reward


class Patient:
    def __init__(self, identification, pre_operating_time, peri_operating_time, post_operating_time):
        self.id = identification
        self.block = False
        self.service_condition = np.array([0, 0, 0])
        self.dismissed = False
        if self.service_condition[2] == 1:
            self.dismissed = True
        self.assigned_resource = None
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

class Resource:
    def __init__(self, identification: Tuple, rate=1):
        # resource id is a tuple: (stage number, resource number)
        self.id = identification
        self.working_status = False
        self.block = False
        self.rate = rate
        self.job_under_process = None
        # Lists to keep track of patients processed at each stage
        self.done_jobs_sequence_pre = list()  # Patients who completed pre-operative stage
        self.done_jobs_sequence_peri = list() # Patients who completed peri-operative stage
        self.done_jobs_sequence_post = list() # Patients who completed post-operative stage

        # self.done_jobs_sequence = list()
        self.total_processing_time = 0

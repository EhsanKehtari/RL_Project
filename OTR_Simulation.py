from typing import Tuple

import numpy as np
from numpy import ndarray


class Job:
    def __init__(self, identification: int, operating_times: ndarray):
        """
        Initialize job.
        :param identification: job's id
        :param operating_times: collection of job's operating times
        """
        self.id = identification
        self.operating_times = operating_times
        self.block = False
        self.service_condition = np.zeros_like(operating_times, dtype=int)
        self.dismissed = False
        self.assigned_resource = None
        self.current_stage_number = 0
        # A dict to contain start and end times at each stage
        self.schedule_dict = dict()
        for stage_num in range(1, len(self.operating_times) + 1):
            self.schedule_dict[stage_num] = np.array([0, 0])


class Resource:
    def __init__(self, identification: Tuple, rate=1):
        """
        Initialize resource.
        :param identification: resource's id (stage number, resource number)
        :param rate: resource's working speed
        """
        self.id = identification
        self.working_status = False
        self.block = False
        self.rate = rate
        self.job_under_process = None
        self.done_jobs_sequence = list()


class Simulators:
    """
    Implement different flexible flowshop simulators.
    List of simulators:
    - LS Simulator
    -
    """

    def __init__(
            self,
            job_machine_matrix: ndarray,
            jobs: ndarray,
            stage_num_name_dict: dict,
            stage_machines_dict: dict,
            jobs_sequence: ndarray
    ):
        """
        Initialize simulators.
        :param job_machine_matrix: matrix of process times of jobs on machines;
                                   shape: (number of jobs, number of machines)
        :param jobs: jobs' dedicated numbers (in order of appearance in job_machine_matrix).
        :param stage_num_name_dict: a dict with stage numbers as keys and stage names as values.
        :param stage_machines_dict: a dict with stage numbers as keys and number of machines at each stage as values.
        :param jobs_sequence: a sequence of jobs to be simulated.
        """
        # Necessary attributes
        self.job_machine_matrix = job_machine_matrix
        self.jobs = jobs
        self.stage_num_name_dict = stage_num_name_dict
        self.stage_machines_dict = stage_machines_dict
        self.jobs_sequence = jobs_sequence
        self.number_of_stages = self.job_machine_matrix.shape[1]
        # Initialize FEL and clock
        self.future_event_list = list()
        self.clock = 0.0
        # To store the information of idle resources and waiting jobs at each stage
        self.info = dict()
        # Initialize resources' instances
        for stage in list(self.stage_machines_dict.keys()):
            self.info[stage] = dict()
            self.info[stage]['Idle Resources'] = list()
            self.info[stage]['Waiting Jobs'] = list()
            # All resources are idle at the beginning of the simulation
            for resource_num in range(1, self.stage_machines_dict[stage] + 1):
                resource = Resource(identification=(stage, resource_num))
                self.info[stage]['Idle Resources'].append(resource)
        # Initialize jobs' resources;
        # All jobs are considered waiting jobs for stage 1 at the beginning
        for job_num in range(len(self.jobs_sequence)):
            job_id = self.jobs_sequence[job_num]
            job_operating_times = self.job_machine_matrix[int(np.argwhere(self.jobs == job_id))]
            job = Job(
                identification=job_id,
                operating_times=job_operating_times
            )
            self.info[1]['Waiting Jobs'].append(job)
        # For each stage's final sequence
        self.stage_final_sequence_dict = dict()
        for stage_num in range(1, self.number_of_stages + 1):
            self.stage_final_sequence_dict[stage_num] = list()
        # Stage 1's final sequence is equal to job_sequence
        self.stage_final_sequence_dict[1] = list(self.jobs_sequence)

    def schedule_job(self, job: object, resource: object, current_clock: float) -> None:
        """
        Schedule the specified job on the specified resource and put it in FEL.
        :param job: the job to be considered for scheduling
        :param resource: the resource on which the job is scheduled on
        :param current_clock: current simulation clock
        :return: None
        """
        # Modify job's attributes
        job.block = False
        job.current_stage_number = resource.id[0]
        # Modify job's start and end time in the specified stage
        job.schedule_dict[job.current_stage_number][0] = current_clock
        job.schedule_dict[job.current_stage_number][1] = \
            current_clock + resource.rate * job.operating_times[job.current_stage_number - 1]
        # Remove job from current stage's waiting jobs list
        self.info[job.current_stage_number]['Waiting Jobs'].remove(job)
        # Modify resource's attributes
        resource.working_status = True
        resource.block = False
        resource.job_under_process = job
        # Add previous resource to previous stage's idle resources list
        if 1 < resource.id[0]:
            self.info[resource.id[0] - 1]['Idle Resources'].append(job.assigned_resource)
        # Remove current resource from current stage's idle resources list
        self.info[resource.id[0]]['Idle Resources'].remove(resource)
        # Modify job's current assigned resource
        job.assigned_resource = resource
        # Determine event type based on stage number
        event_type = 'End of ' + str(self.stage_num_name_dict[resource.id[0]])
        # Schedule job on resource
        event_time = job.schedule_dict[job.current_stage_number][1]
        self.future_event_list.append(
            {
                'Event Type': event_type,
                'Event Time': event_time,
                'Job': job,
                'Resource': resource
            }
        )

    def ls_simulator(self) -> float:
        """
        Simulate the problem based on LS Algorithm.
        :return: makespan (c_max) obtained from simulation of the given sequence
        """
        # Fill future event list with primary events
        while (len(self.info[1]['Waiting Jobs']) != 0) and (len(self.info[1]['Idle Resources']) != 0):
            self.schedule_job(
                job=self.info[1]['Waiting Jobs'][0],
                resource=self.info[1]['Idle Resources'][0],
                current_clock=self.clock
            )
        while len(self.future_event_list) != 0:
            # Sort fel based on event times
            sorted_fel = sorted(self.future_event_list, key=lambda x: x['Event Time'])
            # Find imminent event
            current_event = sorted_fel[0]
            # Restore current event's info from current_event dict
            self.clock = current_event['Event Time']
            current_event_type = current_event['Event Type']
            current_event_job = current_event['Job']
            current_event_resource = current_event['Resource']
            # Execute event (forward look);
            # Modify job's attributes
            current_event_job.block = True
            current_event_job.service_condition[current_event_resource.id[0] - 1] = 1
            # Move job to next stage's waiting jobs list (except for last stage)
            if current_event_resource.id[0] < self.number_of_stages:
                self.info[current_event_resource.id[0] + 1]['Waiting Jobs'].append(current_event_job)
            # Modify resource's attribute
            current_event_resource.working_status = False
            current_event_resource.block = True
            current_event_resource.job_under_process = None
            current_event_resource.done_jobs_sequence.append(current_event_job)
            # Move resource from last stage to last stage's idle resources list (if applicable)
            if current_event_resource.id[0] == self.number_of_stages:
                self.info[current_event_resource]['Idle Resources'].append(current_event_resource)
            # Update current stage's final sequence
            self.stage_final_sequence_dict[current_event_resource.id[0]].append(current_event_job)
            # Remove current event from fel
            self.future_event_list.remove(current_event)
            # Execute event (backward look);
            for stage_num in range(1, self.number_of_stages + 1):
                # Check if any job-resource match can occur
                if len(self.info[stage_num]['Idle Resources']) != 0 and \
                        len(self.info[stage_num]['Waiting Jobs']) != 0:
                    self.schedule_job(
                        job=self.info[stage_num]['Waiting Jobs'][0],
                        resource=self.info[stage_num]['Idle Resources'][0],
                        current_clock=self.clock
                    )
        c_max = self.clock
        return c_max

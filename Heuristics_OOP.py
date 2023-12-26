from typing import Tuple

import numpy as np
from numpy import ndarray


class Flowshop_helper:
    """
    Implement useful helper functions used in class Flowshop.
    List of methods:
    - next_machine_scheduler (static)
    - sequence alternatives (static)
    - c_max_calculator (static)
    """

    def __init__(self):
        return

    @staticmethod
    def next_machine_scheduler(current_schedule: ndarray, process_time: ndarray) -> Tuple[ndarray, None]:
        """
        Schedule jobs on the next machine in a flowshop problem
        :param current_schedule: end times of jobs on the current machine; shape: (number of jobs,)
        :param process_time: process times of jobs; shape: (number of jobs,)
        :return: end times of jobs on the next machine & c_max
        """
        next_schedule = np.zeros_like(current_schedule)
        next_schedule[0] = current_schedule[0] + process_time[0]

        for j in range(1, len(current_schedule)):
            next_schedule[j] = max(current_schedule[j], next_schedule[j - 1]) + process_time[j]
        c_max = next_schedule[-1]
        return next_schedule, c_max

    @staticmethod
    def sequence_alternatives(sequence: ndarray, added_job: any) -> ndarray:
        """
        Give all possible sequence alternatives of given jobs.
        :param sequence: sequence of initial jobs; shape: (number of jobs,)
        :param added_job: job to be placed among initial jobs in the sequence
        :return: new sequence of jobs
        """
        added_sequence = np.zeros((sequence.shape[0] + 1, sequence.shape[0] + 1), dtype=int)

        for row in range(sequence.shape[0] + 1):
            for column in range(sequence.shape[0] + 1):
                if row > column:
                    added_sequence[row][column] = sequence[column]
                elif row < column:
                    added_sequence[row][column] = sequence[column - 1]
                else:
                    added_sequence[row][column] = added_job
        return added_sequence

    @staticmethod
    def c_max_calculator(sequence: ndarray, process_time_matrix: ndarray) -> None:
        """
        Calculate c_max for a sequence of jobs.
        :param sequence: sequence of jobs; shape: (number of jobs,)
        :param process_time_matrix: process times of jobs; shape: (number of jobs, number of machines)
        :return: c_max
        """
        machine_number = process_time_matrix.shape[1]
        current_schedule = np.zeros_like(sequence, dtype=float)

        for machine in range(machine_number):
            current_schedule, c_max = Flowshop_helper.next_machine_scheduler(current_schedule,
                                                                             process_time_matrix[:, machine])
        return c_max


class Flowshop:
    """
    Implement well-known heuristics in flowshop problem.
    List of methods:
    - johnson
    - cds
    - neh
    - mst
    - mot
    - sespt_lespt
    - seopt_leopt
    - lespt_sespt
    - leopt_seopt
    - sespt
    - seopt
    - lespt
    - leopt
    """

    def __init__(self, job_machine_matrix: ndarray, jobs: ndarray):
        """
        :param job_machine_matrix: matrix of process times of jobs on machines;
                                   shape: (number of jobs, number of machines)
        :param jobs: jobs with associated numbers; shape: (number of jobs,)
        """
        self.job_machine_matrix = job_machine_matrix
        self.jobs = jobs

    def johnson(self) -> Tuple[ndarray, None]:
        """
        Implement johnson algorithm on a given set of jobs for two machines.
        :return: optimal job sequence and c_max
        """

        job_machine_matrix_min_column_locator = self.job_machine_matrix.argmin(axis=1)
        job_sorted = self.job_machine_matrix.min(axis=1).argsort()

        right_indicator = job_machine_matrix_min_column_locator[job_sorted]

        right_sequence = np.flip(job_sorted[right_indicator == 1])
        left_sequence = job_sorted[right_indicator == 0]

        sequence = np.concatenate([left_sequence, right_sequence])
        jobs_sequence = self.jobs[sequence]
        c_max = Flowshop_helper.c_max_calculator(sequence, self.job_machine_matrix[list(sequence), :])
        return jobs_sequence

    def cds(self) -> Tuple[ndarray, None]:
        """
        Implement cds algorithm on a given set of jobs and machines.
        :return: best job sequence found and its corresponding c_max
        """

        job_number = self.job_machine_matrix.shape[0]
        machine_number = self.job_machine_matrix.shape[1]

        best_c_max = self.job_machine_matrix.sum()
        best_sequence = 0

        johnson_job_machine_matrix = np.zeros((job_number, 2))

        if machine_number != 1:
            for k in range(1, machine_number):

                machine1 = self.job_machine_matrix[:, 0:k].sum(axis=1)
                machine2 = self.job_machine_matrix[:, machine_number - k:].sum(axis=1)

                johnson_job_machine_matrix[:, 0], johnson_job_machine_matrix[:, 1] = machine1, machine2
                cds_johnson_instance = Flowshop(johnson_job_machine_matrix, np.arange(0, job_number))
                current_sequence = cds_johnson_instance.johnson()
                process_time_machine1 = johnson_job_machine_matrix[:, 0][current_sequence]

                current_schedule, _ = Flowshop_helper().next_machine_scheduler(np.zeros_like(current_sequence, dtype=float),
                                                                        process_time_machine1)
                process_time_machine2 = johnson_job_machine_matrix[:, 1][current_sequence]

                _, current_c_max = Flowshop_helper().next_machine_scheduler(current_schedule, process_time_machine2)

                if current_c_max < best_c_max:
                    best_c_max = current_c_max
                    best_sequence = current_sequence
            jobs_best_sequence = self.jobs[best_sequence]

        else:
            cds_johnson_instance = Flowshop(self.job_machine_matrix, self.jobs)
            jobs_best_sequence = cds_johnson_instance.johnson()

        return jobs_best_sequence

    def neh(self) -> Tuple[ndarray, None]:
        """
        Implement neh algorithm on a given set of jobs and machines.
        :return: best job sequence found and its corresponding c_max
        """

        job_descending_sort = np.flip(self.job_machine_matrix.sum(axis=1).argsort())
        sequence = np.array([job_descending_sort[0]])
        job_number = self.job_machine_matrix.shape[0]

        for iteration in range(1, job_number):

            added_job = job_descending_sort[iteration]
            all_sequence = Flowshop_helper().sequence_alternatives(sequence, added_job)

            best_c_max = self.job_machine_matrix.sum()
            best_sequence = 0

            for possible_sequence in range(all_sequence.shape[0]):

                current_sequence = all_sequence[possible_sequence]
                current_c_max = Flowshop_helper().c_max_calculator(current_sequence,
                                                                   self.job_machine_matrix[list(current_sequence), :])
                if current_c_max < best_c_max:
                    best_c_max = current_c_max
                    best_sequence = current_sequence

            sequence = best_sequence

        jobs_best_sequence = self.jobs[best_sequence]
        return jobs_best_sequence

    def mst(self) -> ndarray:
        """
        Implement mst (mixed surgery time) algorithm.
        :return: sequence of jobs
        """

        process_time_ascending_sort = np.argsort(self.job_machine_matrix)

        jobs_ascending_sort = self.jobs[process_time_ascending_sort]
        jobs_descending_sort = np.flip(jobs_ascending_sort)

        sequence = np.zeros_like(self.jobs)

        for job in range(len(self.jobs)):

            if job % 2 == 0:
                sequence[job] = jobs_ascending_sort[int(np.floor(job / 2))]
            else:
                sequence[job] = jobs_descending_sort[int(np.floor(job / 2))]
        return sequence

    def mot(self) -> ndarray:
        """
        Implement mot (mixed operating time) algorithm.
        :return: sequence of jobs
        """

        process_time_ascending_sort = np.argsort(self.job_machine_matrix)

        jobs_ascending_sort = self.jobs[process_time_ascending_sort]
        jobs_descending_sort = np.flip(jobs_ascending_sort)

        sequence = np.zeros_like(self.jobs)

        for job in range(len(self.jobs)):

            if job % 2 == 0:
                sequence[job] = jobs_ascending_sort[int(np.floor(job / 2))]
            else:
                sequence[job] = jobs_descending_sort[int(np.floor(job / 2))]
        return sequence

    def sespt_lespt(self) -> ndarray:
        """
        Implement sespt_lespt (half increasing in surgery time and half decreasing in surgery time) algorithm.
        :return: sequence of jobs
        """

        process_time_sort = np.argsort(self.job_machine_matrix)

        jobs_sort = self.jobs[process_time_sort]

        sequence = np.zeros_like(self.jobs)

        for job in range(len(self.jobs)):

            if job % 2 == 0:
                sequence[int(np.floor(job / 2))] = jobs_sort[job]
            else:
                sequence[-1 - int(np.floor(job / 2))] = jobs_sort[job]
        return sequence

    def seopt_leopt(self) -> ndarray:
        """
        Implement seopt_leopt (half increasing in operating time and half decreasing in operating time) algorithm.
        :return: sequence of jobs
        """

        process_time_sort = np.argsort(self.job_machine_matrix)

        jobs_sort = self.jobs[process_time_sort]

        sequence = np.zeros_like(self.jobs)

        for job in range(len(self.jobs)):

            if job % 2 == 0:
                sequence[int(np.floor(job / 2))] = jobs_sort[job]
            else:
                sequence[-1 - int(np.floor(job / 2))] = jobs_sort[job]
        return sequence

    def lespt_sespt(self) -> ndarray:
        """
        Implement lespt_sespt (half decreasing in surgery time and half increasing in surgery time) algorithm.
        :return: sequence of jobs
        """

        process_time_sort = np.argsort(self.job_machine_matrix)
        process_time_sort = np.flip(process_time_sort)

        jobs_sort = self.jobs[process_time_sort]

        sequence = np.zeros_like(self.jobs)

        for job in range(len(self.jobs)):

            if job % 2 == 0:
                sequence[int(np.floor(job / 2))] = jobs_sort[job]
            else:
                sequence[-1 - int(np.floor(job / 2))] = jobs_sort[job]
        return sequence

    def leopt_seopt(self) -> ndarray:
        """
        Implement leopt_seopt (half decreasing in operating time and half increasing in operating time) algorithm.
        :return: sequence of jobs
        """

        process_time_sort = np.argsort(self.job_machine_matrix)
        process_time_sort = np.flip(process_time_sort)

        jobs_sort = self.jobs[process_time_sort]

        sequence = np.zeros_like(self.jobs)

        for job in range(len(self.jobs)):

            if job % 2 == 0:
                sequence[int(np.floor(job / 2))] = jobs_sort[job]
            else:
                sequence[-1 - int(np.floor(job / 2))] = jobs_sort[job]
        return sequence

    def sespt(self) -> ndarray:
        """
        Implement sespt (shortest expected surgery processing time) algorithm.
        :return: sequence of jobs
        """

        process_time_sort = np.argsort(self.job_machine_matrix)
        return self.jobs[process_time_sort]

    def seopt(self) -> ndarray:
        """
        Implement seopt (shortest expected OTR processing time) algorithm.
        :return: sequence of jobs
        """

        process_time_sort = np.argsort(self.job_machine_matrix)
        return self.jobs[process_time_sort]

    def lespt(self) -> ndarray:
        """
        Implement lespt (longest expected surgery processing time) algorithm.
        :return: sequence of jobs
        """

        process_time_sort = np.argsort(self.job_machine_matrix)
        process_time_sort = np.flip(process_time_sort)
        return self.jobs[process_time_sort]

    def leopt(self) -> ndarray:
        """
        Implement leopt (longest expected OTR processing time) algorithm.
        :return: sequence of jobs
        """

        process_time_sort = np.argsort(self.job_machine_matrix)
        process_time_sort = np.flip(process_time_sort)
        return self.jobs[process_time_sort]

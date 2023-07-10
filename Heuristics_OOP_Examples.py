import numpy as np
from Heuristics_OOP import Flowshop

# johnson
test_johnson_job_machine_matrix = np.array(
    [[75.903, 85.933],
    [102.2, 93.3],
    [29.73, 26],
    [58.137, 101],
    [84.92, 107]]
)
test_johnson_jobs = np.array(
    [1, 2, 3, 4, 5]
)
test_johnson_instance = Flowshop(test_johnson_job_machine_matrix, test_johnson_jobs)
test_johnson_result = test_johnson_instance.johnson()
print(test_johnson_result)

# cds
test_cds_job_machine_matrix = np.array(
    [[75.903, 85.933],
    [102.2, 93.3],
    [29.73, 26],
    [58.137, 101],
    [84.92, 107]]
)
test_cds_jobs = np.array(
    [1, 2, 3, 4, 5]
)
test_cds_instance = Flowshop(test_cds_job_machine_matrix, test_cds_jobs)
test_cds_result = test_cds_instance.cds()
print(test_cds_result)

# neh
test_neh_job_machine_matrix = np.array(
    [[5.12, 4.4, 7.4, 8.6],
    [3.3, 8.0, 8.8, 4.6],
    [6.1, 2.1, 7.2, 2.5],
    [4.0, 9.9, 6.2, 9.4],
    [9.1, 13.1, 5.2, 1.1]]
)
test_neh_jobs = np.array(
    [1, 2, 3, 4, 5]
)
test_neh_instance = Flowshop(test_neh_job_machine_matrix, test_neh_jobs)
test_neh_result = test_neh_instance.neh()
print(test_neh_result)

# mst
test_mst_job_machine_matrix = np.array(
    [75.903, 102.2, 29.73, 58.137, 84.92]
)
test_mst_jobs = np.array(
    [10, 6, 5, 9, 11]
)
test_mst_instance = Flowshop(test_mst_job_machine_matrix, test_mst_jobs)
test_mst_result = test_mst_instance.mst()
print(test_mst_result)

# mot
test_mot_job_machine_matrix = np.array(
    [75.903, 102.2, 29.73, 58.137, 84.92]
)
test_mot_jobs = np.array(
    [10, 6, 5, 9, 11]
)
test_mot_instance = Flowshop(test_mot_job_machine_matrix, test_mot_jobs)
test_mot_result = test_mot_instance.mot()
print(test_mot_result)

# sespt_lespt
test_sespt_lespt_job_machine_matrix = np.array(
    [75.903, 102.2, 29.73, 58.137, 84.92]
)
test_sespt_lespt_jobs = np.array(
    [10, 6, 5, 9, 11]
)
test_sespt_lespt_instance = Flowshop(test_sespt_lespt_job_machine_matrix, test_sespt_lespt_jobs)
test_sespt_lespt_result = test_sespt_lespt_instance.sespt_lespt()
print(test_sespt_lespt_result)

# seopt_leopt
test_seopt_leopt_job_machine_matrix = np.array(
    [75.903, 102.2, 29.73, 58.137, 84.92]
)
test_seopt_leopt_jobs = np.array(
    [10, 6, 5, 9, 11]
)
test_seopt_leopt_instance = Flowshop(test_seopt_leopt_job_machine_matrix, test_seopt_leopt_jobs)
test_seopt_leopt_result = test_seopt_leopt_instance.seopt_leopt()
print(test_seopt_leopt_result)

# lespt_sespt
test_lespt_sespt_job_machine_matrix = np.array(
    [75.903, 102.2, 29.73, 58.137, 84.92]
)
test_lespt_sespt_jobs = np.array(
    [10, 6, 5, 9, 11]
)
test_lespt_sespt_instance = Flowshop(test_lespt_sespt_job_machine_matrix, test_lespt_sespt_jobs)
test_lespt_sespt_result = test_lespt_sespt_instance.lespt_sespt()
print(test_lespt_sespt_result)

# leopt_seopt
test_leopt_seopt_job_machine_matrix = np.array(
    [75.903, 102.2, 29.73, 58.137, 84.92]
)
test_leopt_seopt_jobs = np.array(
    [10, 6, 5, 9, 11]
)
test_leopt_seopt_instance = Flowshop(test_leopt_seopt_job_machine_matrix, test_leopt_seopt_jobs)
test_leopt_seopt_result = test_leopt_seopt_instance.leopt_seopt()
print(test_leopt_seopt_result)

# sespt
test_sespt_job_machine_matrix = np.array(
    [7.5, 3, 2.5, 6.3, 5.4, 2.22]
)
test_sespt_jobs = np.array(
    [2, 5, 6, 11, 17, 25]
)
test_sespt_instance = Flowshop(test_sespt_job_machine_matrix, test_sespt_jobs)
test_sespt_result = test_sespt_instance.sespt()
print(test_sespt_result)

# seopt
test_seopt_job_machine_matrix = np.array(
    [7.5, 3, 2.5, 6.3, 5.4, 2.22]
)
test_seopt_jobs = np.array(
    [2, 5, 6, 11, 17, 25]
)
test_seopt_instance = Flowshop(test_seopt_job_machine_matrix, test_seopt_jobs)
test_seopt_result = test_seopt_instance.seopt()
print(test_seopt_result)

# lespt
test_lespt_job_machine_matrix = np.array(
    [7.5, 3, 2.5, 6.3, 5.4, 2.22]
)
test_lespt_jobs = np.array(
    [2, 5, 6, 11, 17, 25]
)
test_lespt_instance = Flowshop(test_lespt_job_machine_matrix, test_lespt_jobs)
test_lespt_result = test_lespt_instance.lespt()
print(test_lespt_result)

# leopt
test_leopt_job_machine_matrix = np.array(
    [7.5, 3, 2.5, 6.3, 5.4, 2.22]
)
test_leopt_jobs = np.array(
    [2, 5, 6, 11, 17, 25]
)
test_leopt_instance = Flowshop(test_leopt_job_machine_matrix, test_leopt_jobs)
test_leopt_result = test_leopt_instance.leopt()
print(test_leopt_result)
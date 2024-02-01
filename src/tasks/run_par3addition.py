# # https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
#
# from src import model_task_handler
# from matplotlib import pyplot as plt
# from ..models import MODELS, model_to_module
#
#
# model_module = model_to_module(MODELS.PAR3ADD)
#
#
# if __name__ == '__main__':
#     TASKS = [(MODELS.PAR3ADD, {"points_per_second": 0.1, "tL": 1600, "Nx":100,
#                "initial_condition":[0.1]*100+[0.5]*100+[0]*30+[0]*70,
#                # "D_A": 0.35, "D_J": 0.1,
#                "rho_A": 0.3, "rho_J": 0.3, "rho_P": 2,
#                # "sigma_P": 0,
#                "k_PA": 2, "k_AJ": 2
#            })]
#
#     results = model_task_handler.run_tasks_parallel(TASKS)
#
#     for res in results:
#         if res[1] == "FAILURE":
#             continue
#
#         model_module.animate_plot(res[1], res[2], save_file=False, rescale=False)
#
#     plt.show()
#

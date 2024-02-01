# # https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
#
# import matplotlib
# import numpy as np
# from src.models.metric_functions import polarity_get_all
# from src import model_task_handler
# from matplotlib import pyplot as plt
# from ..models import MODELS, model_to_module
#
#
# model_module = model_to_module(MODELS.PAR3ADD)
#
#
# # range_info is of form [start, stop, steps]
# def get_parameter_space_tasks(def_params, first_key, first_range, second_key, second_range):
#     task_list = []
#
#     for v1 in np.linspace(first_range[0], first_range[1], first_range[2]):
#         for v2 in np.linspace(second_range[0], second_range[1], second_range[2]):
#             task_list.append(
#                 (MODELS.PAR3ADD, {**def_params, first_key: v1, second_key: v2})
#             )
#     return task_list
#
#
# if __name__ == '__main__':
#     Nx = 100
#     tL = 6000
#
#     def_params = {
#         "label": "par3addition",
#         "points_per_second": 0.01,
#
#         "Nx": Nx,
#         "tL": tL,
#     }
#
#     key1 = "k_JP"
#     range1 = [0, 4, 5]
#
#     key2 = "k_AJ"
#     range2 = [0, 4, 5]
#
#     tasks = get_parameter_space_tasks(def_params, key1, range1, key2, range2)
#     results = model_task_handler.run_tasks_parallel(tasks)
#
#     # make scatter plot
#     # colour for m
#     plt.figure()
#     cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["grey", "yellow", "violet", "blue"])
#     norm = plt.Normalize(0, 1)
#
#     for res in results:
#         sol = res[1]
#         kvals = res[2]
#
#         # check failure
#         if sol == "FAILURE":
#             plt.scatter(kvals[key1], kvals[key2], marker="x", s=100)
#             continue
#
#         p, _, marker = polarity_get_all(kvals["X"], sol.y[kvals["Nx"]:2*kvals["Nx"], -1], sol.y[2*kvals["Nx"]:, -1], kvals["Nx"])
#         alpha = np.minimum(1, p + 0.2)
#         plt.scatter(kvals[key1], kvals[key2], c=p, cmap=cmap, norm=norm, alpha=alpha, marker=marker, s=100)
#
#     clb = plt.colorbar()
#     clb.ax.set_title("polarity")
#
#     # plt.xlabel("peak position")
#     # plt.ylabel("time before dropping")
#     # plt.title("goehring v_function variation Nx=" + str(Nx) + ",tL=" + str(tL))
#
#     plt.show()
#

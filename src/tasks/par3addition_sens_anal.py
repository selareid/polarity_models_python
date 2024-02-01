# # https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
# import os
#
# from src.models.metric_functions import polarity_measure
#
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
#
# import numpy as np
# from matplotlib import pyplot as plt
# from src.tasks.variation_task_helper import generate_tasks, split_baseline_from_results, run_grouped_tasks, \
#     get_variation_multiplier, get_xticks
# from ..models import MODELS, model_to_module
#
# model_module = model_to_module(MODELS.PAR3ADD)
#
# def get_default_parameters(Nx=50, tL=3000):
#     return {
#     "label": "par3addition",
#     "points_per_second": 0.01,
#
#     # General Setup Variables
#     "Nx": Nx,  # number of length steps
#     "L": 134.6,  # length of region
#     "x0": 0,
#     "xL": 67.3,  # L / 2
#     "t0": 0,
#     "tL": tL,
#
#     # Model parameters and functions #
#     "psi": 0.174,  # surface to volume conversion factor
#
#     # diffusion coefficients
#     "D_J": 0.28,
#     "D_A": 0.28,
#     "D_P": 0.15,
#
#     # velocity coefficients
#     "sigma_J": 1,
#     "sigma_P": 1,
#
#     "k_onJ": 8.58 * 10 ** (-3),  # value taken from A equivalent
#     "k_onP": 4.74 * 10 ** (-2),
#
#     "k_offA": 5.4 * 10 ** (-3),
#     "k_offP": 7.3 * 10 ** (-3),
#
#     "k_JP": 0.190,  # values taken from k_AP
#     "k_AJ": 0.190,  # values taken from k_AP
#     "k_PA": 2.0,
#
#     "rho_J": 1.56,
#     "rho_A": 1.56,
#     "rho_P": 1.0,
#
#     "gamma": 1,
#     "alpha": 1,
#     "beta": 2,
#
#     # R_X
#     # Xbar
#     # "J_cyto": default_J_cyto,
#     # "A_cyto": default_A_cyto,
#     # "P_cyto": default_P_cyto,
#     # "v_func": default_v_func,
# }
#
#
# def test_polarity_establishment(Nx=50,tL=3000):
#     # initial_condition = [0]*(Nx//3) + [2.828] * (Nx - Nx//3) + [0.025] * Nx
#     initial_condition = [1] * Nx + [1]*Nx + [0.025] * Nx  # the values we get when running baseline with v=0
#
#     variation_multipliers, index_for_100x = get_variation_multiplier("sparse")
#     xticks = get_xticks(variation_multipliers, index_for_100x)
#
#     tasks = generate_tasks(MODELS.PAR3ADD, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL),
#                            {"label": ".", "initial_condition": initial_condition})
#
#     print(f"Testing polarity establishment")
#     print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")
#
#     baseline, results_by_variable = split_baseline_from_results(MODELS.PAR3ADD, run_grouped_tasks(tasks), index_for_100x)
#
#     print("Plotting")
#     model_module.plot_variation_sets(results_by_variable, label="Par3Add Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks)
#     # model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks, xlim=["60%", "140%"])
#
#     # Get results with >0.5 polarity
#     # has form [{key:key, value:value}]
#     good_res = []
#     for variation in results_by_variable:
#         sol_list = variation[0]
#         kvals_list = variation[1]
#
#         for i in np.arange(0, len(sol_list)):
#             sol = sol_list[i]
#             kvals = kvals_list[i]
#
#             if not sol == "FAILURE" and polarity_measure(kvals["X"], sol.y[kvals["Nx"]:2*kvals["Nx"], -1], sol.y[2*kvals["Nx"]:, -1], kvals["Nx"]) > 0.5:
#                 key_varied = kvals["key_varied"]
#                 good_res.append({"key": key_varied, "value": kvals[key_varied]})
#
#     return good_res
#
#
# # def run_again(Nx=50,tL=3000, key=None, value=None):
# #     # initial_condition = [0]*(Nx//3) + [2.828] * (Nx - Nx//3) + [0.025] * Nx
# #     initial_condition = [1] * Nx + [1]*Nx + [0.025] * Nx  # the values we get when running baseline with v=0
# #
# #     variation_multipliers, index_for_100x = get_variation_multiplier("sparse")
# #     xticks = get_xticks(variation_multipliers, index_for_100x)
# #
# #     defs = get_default_parameters(Nx=Nx, tL=tL)
# #     defs[key] = value
# #
# #     tasks = generate_tasks(MODELS.PAR3ADD, variation_multipliers, defs,
# #                            {"label": format(f"{key}.{value}"), "initial_condition": initial_condition})
# #
# #     print(f"Testing polarity establishment")
# #     print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")
# #
# #     baseline, results_by_variable = split_baseline_from_results(MODELS.PAR3ADD, run_grouped_tasks(tasks), index_for_100x)
# #
# #     print("Plotting")
# #     model_module.plot_variation_sets(results_by_variable, label=format(f"Par3Add Again {key}={value},Nx={str(Nx)},tL={str(tL)}"), x_axis_labels=xticks)
#
#
#
# if __name__ == '__main__':
#     more_polarised = test_polarity_establishment(Nx=10)
#
#     # for k,v in more_polarised:
#     #     print("")
#     #     run_again(Nx=10, key=k, value=v)
#
#
#
#     print("Finished!")
#     plt.show()
#

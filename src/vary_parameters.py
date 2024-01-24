import time
# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
from matplotlib import pyplot as plt
from models import MODELS, model_to_module
import model_task_handler

def a_func(kvals, lt, x): return 0
def v_func(kvals, x, t):
    # return np.minimum(0.7, np.maximum(0, x - 35)) / (np.max([t, 100]) - 99) ** 3
    # return -x*(x-67.3)*(x-30) / (50000*(np.maximum(1,np.abs((t-200)/150)))**2)
    return 0

# task format is tuples like (model, args) in a list
# TASKS = [(MODELS.TOSTEVIN, {"points_per_second": 1, "tL": 1000, "sort": 0.25, "Nx": 100}),
#     (MODELS.TOSTEVIN, {"label": "Dm=0.1", "points_per_second": 1, "tL": 1000, "Dm": 0.1, "sort": 0.1, "Nx": 100}),
#     (MODELS.TOSTEVIN, {"label": "Dm=0.2", "points_per_second": 1, "tL": 1000, "Dm": 0.2, "sort": 0.2, "Nx": 100}),
#     (MODELS.TOSTEVIN, {"label": "Dm=0.4", "points_per_second": 1, "tL": 1000, "Dm": 0.4, "sort": 0.4, "Nx": 100}),
#     (MODELS.TOSTEVIN, {"label": "Dm=0.5", "points_per_second": 1, "tL": 1000, "Dm": 0.5, "sort": 0.5, "Nx": 100}),
#     (MODELS.TOSTEVIN, {"label": "Dm=1", "points_per_second": 1, "tL": 1000, "Dm": 1, "sort": 1, "Nx": 100}),
#     (MODELS.TOSTEVIN, {"label": "Dm=2.5", "points_per_second": 1, "tL": 1000, "Dm": 2.5, "sort": 2.5, "Nx": 100}),
#     (MODELS.TOSTEVIN, {"label": "Dm=5", "points_per_second": 1, "tL": 1000, "Dm": 5, "sort": 5, "Nx": 100}),
#     (MODELS.TOSTEVIN, {"label": "Dm=10", "points_per_second": 1, "tL": 1000, "Dm": 10, "sort": 10, "Nx": 100}),]

TASKS = [(MODELS.GOEHRING, {"points_per_second": 1, "tL": 1000, "Nx": 300}),]

# TASKS = [(MODELS.GOEHRING, {"points_per_second": 1, "tL": 3600, "v_func": v_func}),]

# TASKS = [(MODELS.GOEHRING, {"label": "Dp=0.5", "points_per_second": 1, "tL": 300, "D_P": 0.5, "Nx": 300}),
#          (MODELS.GOEHRING, {"label": "Dp=0.2", "points_per_second": 1, "tL": 300, "D_P": 0.2, "Nx": 300}),
#          (MODELS.GOEHRING, {"label": "Dp=0.1", "points_per_second": 1, "tL": 300, "D_P": 0.1, "Nx": 300}),
#          (MODELS.GOEHRING, {"label": "Dp=0.01", "points_per_second": 1, "tL": 300, "D_P": 0.01, "Nx": 300}),]


results = model_task_handler.run_tasks_parallel(TASKS)

results.sort(key=lambda res: 0 if res[1]=="FAILURE" or not "sort" in res[2] else res[2]["sort"])  # order when plotting multiple solutions on single figure

sol_list = []
kvals_list = []

for res in results:
    if res[1] == "FAILURE":
        continue

    model_module = model_to_module(res[0])

    ### Plots and Animated Plots for individual solutions
    model_module.plot_overall_quantities_over_time(res[1], res[2])
    model_module.plot_final_timestep(res[1], res[2])
    model_module.animate_plot(res[1], res[2], save_file=False)

    sol_list.append(res[1])
    kvals_list.append(res[2])

### Below Outputs Final Timestep for All Solutions ###
# it makes some assumptions about the input e.g. all same shape for time/space, that all inputs are relevant, ...
### For tostevin
# model_module = model_to_module(MODELS.TOSTEVIN)
# model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Ac=False,plot_Pm=False,plot_Pc=False)
# model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Am=False,plot_Pm=False,plot_Pc=False)
# model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Am=False,plot_Ac=False,plot_Pc=False)
# model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Am=False,plot_Ac=False,plot_Pm=False)
# model_module.plot_multi_final_timestep(sol_list, kvals_list)
### For goehring
# model_module = model_to_module(MODELS.TOSTEVIN)
# model_module.plot_multi_final_timestep(sol_list, kvals_list)


plt.show()  # if we don't have this the program ends and the plots close

# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src import model_task_handler
from matplotlib import pyplot as plt
from ..models import MODELS, model_to_module


def v_func_zero(_kvals, _x, _t):
    return 0


model_module = model_to_module(MODELS.CRUMBS)

if __name__ == '__main__':
    TASKS = [(MODELS.CRUMBS, {"points_per_second": 0.1, "tL": 500, "Nx":100, "v_func": v_func_zero,
               "initial_condition":[0]*50+[1]*50 + [0]*50+[1]*50 + [1]*50+[0]*50 + [0]*100
           }),]
       #   (MODELS.CRUMBS, {"points_per_second": 0.1, "tL": 1600, "Nx":100,
       #     "initial_condition":[0]*50+[1]*50 + [0]*100 + [4]*30+[4]*70 + [0]*100,
       #     # "D_A": 0.35, "D_J": 0.01,
       #     # "rho_A": 0.7, "rho_J": 0.3,
       #     # "sigma_P": 0,
       #     # "k_PA": 2, "k_AJ": 2, "k_offA": 0.0054
       # })]

    results = model_task_handler.run_tasks_parallel(TASKS)

    for res in results:
        if res[1] == "FAILURE":
            continue

        model_module.animate_plot(res[1], res[2], save_file=False, rescale=False)

    plt.show()


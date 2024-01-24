# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from matplotlib import pyplot as plt
from src.tasks.variation_task_helper import generate_tasks, split_baseline_from_results, run_grouped_tasks, \
    get_variation_multiplier, get_xticks
from ..models import MODELS, model_to_module

model_module = model_to_module(MODELS.GOEHRING)


def default_v_func(kvals, x, t):
    v_time = 600
    time_factor = 1 / np.maximum(1, t / 10 - v_time / 10)

    center = kvals["xL"] / 4
    sd = np.minimum(center / 4, (kvals["xL"] - center) / 4)
    peak = 0.1

    return time_factor * peak * np.exp(-(x - center) ** 2 / (2 * sd ** 2))


def v_func_zero(kvals, x, t):
    return 0


def get_default_parameters(Nx=50, tL=3000, v_func=default_v_func, filter=None):
    if filter == "A": # Just A Stuff
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                "D_A": 0.28,

                "k_onA": 8.58 * 10 ** (-3),

                "k_offA": 5.4 * 10 ** (-3),

                "k_AP": 0.190,

                "rho_A": 1.56,

                "v_func": v_func
         }
    elif filter == "P": # Just P Stuff
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                "D_P": 0.15,

                "k_onP": 4.74 * 10 ** (-2),

                "k_offP": 7.3 * 10 ** (-3),

                "k_PA": 2.0,

                "rho_P": 1.0,

                "v_func": v_func
                }
    elif filter == "generic": # not A, not P stuff
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                "psi": 0.174,

                "alpha": 1,
                "beta": 2,

                "v_func": v_func
                }
    else: # All
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                "psi": 0.174,

                "D_A": 0.28,
                "D_P": 0.15,

                "k_onA": 8.58 * 10 ** (-3),
                "k_onP": 4.74 * 10 ** (-2),

                "k_offA": 5.4 * 10 ** (-3),
                "k_offP": 7.3 * 10 ** (-3),

                "k_AP": 0.190,
                "k_PA": 2.0,

                "rho_A": 1.56,
                "rho_P": 1.0,

                "alpha": 1,
                "beta": 2,

                # R_X
                # Xbar
                # "A_cyto": default_A_cyto,
                # "P_cyto": default_P_cyto,
                "v_func": v_func
                }


def test_polarity_establishment(Nx=50,tL=3000, filter=None):
    # initial_condition = [0]*(Nx//3) + [2.828] * (Nx - Nx//3) + [0.025] * Nx
    initial_condition = [2.828] * Nx + [0.025] * Nx  # the values we get when running baseline with v=0

    variation_multipliers, index_for_100x = get_variation_multiplier()
    xticks = get_xticks(variation_multipliers, index_for_100x)

    tasks = generate_tasks(MODELS.GOEHRING, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL, filter=filter), {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity establishment")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    baseline, results_by_variable = split_baseline_from_results(MODELS.GOEHRING, run_grouped_tasks(tasks), index_for_100x)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks)
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks, xlim=["60%", "140%"])


def test_polarity_maintenance(Nx=50, tL=3000, filter=None):
    initial_condition = [0] * (Nx // 2) + [1] * (Nx - Nx // 2) + [1] * (Nx // 2) + [0] * (Nx - Nx // 2) # polarised

    variation_multipliers, index_for_100x = get_variation_multiplier()
    xticks = get_xticks(variation_multipliers, index_for_100x)

    tasks = generate_tasks(MODELS.GOEHRING, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL, filter=filter, v_func=v_func_zero), {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity maintenance")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    baseline, results_by_variable = split_baseline_from_results(MODELS.GOEHRING, run_grouped_tasks(tasks), index_for_100x)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL), x_axis_labels=xticks)
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL), x_axis_labels=xticks, xlim=["60%", "140%"])


if __name__ == '__main__':
    # test_polarity_establishment(Nx=100, filter="A")
    # test_polarity_establishment(Nx=100, filter="P")
    # test_polarity_establishment(Nx=100, filter="generic")
    # test_polarity_establishment(Nx=100, filter=None)

    test_polarity_maintenance(Nx=100, filter="A")
    test_polarity_maintenance(Nx=100, filter="P")
    test_polarity_maintenance(Nx=100, filter="generic")
    test_polarity_maintenance(Nx=100, filter=None)
    plt.show()


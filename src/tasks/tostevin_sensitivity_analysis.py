# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from matplotlib import pyplot as plt
from src.tasks.variation_task_helper import generate_tasks, run_grouped_tasks, split_baseline_from_results, get_xticks, \
    get_variation_multiplier
from .. import model_task_handler
from ..models import MODELS, model_to_module

model_module = model_to_module(MODELS.TOSTEVIN)


def get_default_parameters(Nx=50, tL=3000, filter=None):
    if filter == "A": #  Just A stuff
        return {
            "key_varied": "",
            "label": "tostevin",
            "points_per_second": 0.01,

            # General Setup Variables
            "Nx": Nx,  # number of length steps
            "L": 50.0,  # length of region - model parameter
            "x0": 0,
            "xL": 50,  # L
            "t0": 0,
            "tL": tL,

            "cA1": 0.01,
            "cA2": 0.07,
            "cA3": 0.01,
            "cA4": 0.11,
        }
    elif filter == "P":  # Just P stuff
        return {
            "key_varied": "",
            "label": "tostevin",
            "points_per_second": 0.01,

            # General Setup Variables
            "Nx": Nx,  # number of length steps
            "L": 50.0,  # length of region - model parameter
            "x0": 0,
            "xL": 50,  # L
            "t0": 0,
            "tL": tL,

            "cP1": 0.08,
            "cP3": 0.04,
            "cP4": 0.13,
        }
    elif filter == "generic":  # Not A, not P stuff
        return {
            "key_varied": "",
            "label": "tostevin",
            "points_per_second": 0.01,

            # General Setup Variables
            "Nx": Nx,  # number of length steps
            "L": 50.0,  # length of region - model parameter
            "x0": 0,
            "xL": 50,  # L
            "t0": 0,
            "tL": tL,

            # Model parameters and functions
            # Taken from pg3 in paper, used for Figs2,3
            "Dm": 0.25,
            "Dc": 5,
            "lambda_0": 42.5,
            "lambda_1": 27,
            "a_0": 1,
            "epsilon": 0.4,
        }
    else:  # All
        return {
            "key_varied": "",
            "label": "tostevin",
            "points_per_second": 0.01,

            # General Setup Variables
            "Nx": Nx,  # number of length steps
            "L": 50.0,  # length of region - model parameter
            "x0": 0,
            "xL": 50,  # L
            "t0": 0,
            "tL": tL,

            # Model parameters and functions
            # Taken from pg3 in paper, used for Figs2,3
            "Dm": 0.25,
            "Dc": 5,
            "cA1": 0.01,
            "cA2": 0.07,
            "cA3": 0.01,
            "cA4": 0.11,
            "cP1": 0.08,
            "cP3": 0.04,
            "cP4": 0.13,
            "lambda_0": 42.5,
            "lambda_1": 27,
            "a_0": 1,
            "epsilon": 0.4,

            # "a_func": default_a_func
        }


def test_polarity_establishment(Nx=50, tL=3000, filter=None, extra_plot=None):
    initial_condition = [1]*Nx + [0]*Nx + [0]*Nx + [1]*Nx \
              + [50]  # L

    variation_multipliers, index_for_100x = get_variation_multiplier()
    xticks = get_xticks(variation_multipliers, index_for_100x)
    print(xticks)

    tasks = generate_tasks(MODELS.TOSTEVIN, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL, filter=filter), {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity establishment")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")
    
    baseline_result, results_by_variable = split_baseline_from_results(MODELS.TOSTEVIN, run_grouped_tasks(tasks), index_for_100x, extra_plot=extra_plot)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Tostevin Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks)
    model_module.plot_variation_sets(results_by_variable, label="Tostevin Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks, xlim=["60%", "140%"])


# def a_func_zero(kvals, lt, x):
#     return 0


# def test_polarity_maintenance(Nx=50,tL=3000):
#     #polarised IC
#     initial_condition = [1]*(Nx//2) + [0]*(Nx-Nx//2) + [0.5]*Nx \
#              + [0]*(Nx//2) + [1]*(Nx-Nx//2) + [0.5]*Nx + [50] # L
#
#     # variation_multipliers = [0.1, 0.25, 0.5, 2, 5, 10] # wide
#     # variation_multipliers = [0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3]  # narrow
#     variation_multipliers = [0.5, 0.6, 0.7, 0.9, 1.2, 1.3, 1.5, 1.75, 2]  # middle
#     index_for_100x = 3
#     xticks = [format(f"{x*100:.4f}%") for x in variation_multipliers]
#     xticks.insert(index_for_100x, "100%")
#
#     tasks = generate_tasks(MODELS.TOSTEVIN, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL), {"label": ".", "initial_condition": initial_condition, "a_func": a_func_zero})
#
#     print(f"Testing polarity maintenance with a=0")
#     print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0, len(tasks))])} tasks")
#
#     baseline_result, results_by_variable = split_baseline_from_results(MODELS.TOSTEVIN, run_grouped_tasks(tasks), index_for_100x)
#
#     print("Plotting")
#     model_module.plot_variation_sets(results_by_variable, label="Tostevin Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks)


if __name__ == '__main__':
    # test_polarity_establishment(Nx=100, filter="A")

    extra_plot_list = []
    for key in get_default_parameters(100, 3000, filter="P"):
        if key not in ["key_varied", "label", "points_per_second", "Nx", "L", "x0", "xL", "t0", "tL", "v_func", "a_func"]:
            extra_plot_list.append(format(f"{key}={get_default_parameters(100, 3000, filter='P')[key] * 10:.2f}"))
            extra_plot_list.append(format(f"{key}={get_default_parameters(100, 3000, filter='P')[key] * 0.1:.2f}"))

    test_polarity_establishment(Nx=100, filter="P", extra_plot=extra_plot_list)
    # test_polarity_establishment(Nx=100, filter="generic")
    # test_polarity_establishment(Nx=100, filter=None)
    # test_polarity_maintenance(Nx=24)

    print("Finished!")
    plt.show()

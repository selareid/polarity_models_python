from matplotlib import pyplot as plt
from src.tasks.variation_task_helper import generate_tasks, run_grouped_tasks, split_baseline_from_results
from .. import model_task_handler
from ..models import MODELS, model_to_module

model_module = model_to_module(MODELS.TOSTEVIN)


def get_default_parameters(Nx=50, tL=3000):
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


def test_polarity_establishment(Nx=50,tL=3000):
    initial_condition = [1]*Nx + [0]*Nx + [0]*Nx + [1]*Nx \
              + [50]  # L

    # # variation_multipliers = [0.5, 0.75, 1.5, 2] # 1x is assumed
    variation_multipliers = [0.1, 0.25, 0.5, 2, 5, 10] # 1x is assumed
    index_for_100x = 3
    xticks = [format(f"{x*100:.2f}%") for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")

    tasks = generate_tasks(MODELS.TOSTEVIN, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL), {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity establishment")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")
    
    baseline_result, results_by_variable = split_baseline_from_results(MODELS.TOSTEVIN, run_grouped_tasks(tasks), index_for_100x)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Tostevin Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks)


def a_func_zero(kvals, lt, x):
    return 0


def test_polarity_maintenance(Nx=50,tL=3000):
    #polarised IC
    initial_condition = [1]*(Nx//2) + [0]*(Nx-Nx//2) + [0.5]*Nx \
             + [0]*(Nx//2) + [1]*(Nx-Nx//2) + [0.5]*Nx + [50] # L

    # variation_multipliers = [0.5, 0.75, 1.5, 2] # 1x is assumed
    variation_multipliers = [0.1, 0.25, 0.5, 2, 5, 10] # 1x is assumed
    index_for_100x = 3
    xticks = [format(f"{x*100:.2f}%") for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")

    tasks = generate_tasks(MODELS.TOSTEVIN, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL), {"label": ".", "initial_condition": initial_condition, "a_func": a_func_zero})

    print(f"Testing polarity maintenance with a=0")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0, len(tasks))])} tasks")

    baseline_result, results_by_variable = split_baseline_from_results(MODELS.TOSTEVIN, run_grouped_tasks(tasks), index_for_100x)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Tostevin Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks)


if __name__ == '__main__':
    test_polarity_establishment(Nx=24)
    # test_polarity_maintenance(Nx=24)
    plt.show()

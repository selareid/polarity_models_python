import numpy as np
from matplotlib import pyplot as plt
from src.tasks.variation_task_helper import generate_tasks, split_baseline_from_results, run_grouped_tasks
from ..models import MODELS, model_to_module

model_module = model_to_module(MODELS.GOEHRING)


def default_v_func(kvals, x, t):
    P = 0.1
    H = kvals["xL"]*0.5/2

    rescale_factor = 1 / np.maximum(1, t/50-300/50)**2

    P = P * rescale_factor

    if x <= H:
        return x*P/H
    else:  # x > H
        return P-P*(x-H)/(67.3-H)


def v_func_zero(kvals, x, t):
    return 0

def get_default_parameters(Nx=50, tL=3000, v_func=default_v_func):
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
        "v_func": v_func,
    }


def test_polarity_establishment(Nx=50,tL=3000):
    # initial_condition = [0]*(Nx//3) + [2.828] * (Nx - Nx//3) + [0.025] * Nx
    initial_condition = [2.828] * Nx + [0.025] * Nx  # the values we get when running baseline with v=0

    # variation_multipliers = [0.1, 0.25, 0.5, 2, 5, 10] # wide
    # variation_multipliers = [0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3]  # narrow
    variation_multipliers = [0.5, 0.6, 0.7, 0.9, 1.2, 1.3, 1.5, 1.75, 2]  # middle
    index_for_100x = 4
    xticks = [format(f"{x * 100:.2f}%") for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")

    tasks = generate_tasks(MODELS.GOEHRING, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL), {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity establishment")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    baseline, results_by_variable = split_baseline_from_results(MODELS.GOEHRING, run_grouped_tasks(tasks), index_for_100x)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks)


def test_polarity_maintenance(Nx=50,tL=3000):
    initial_condition = [0] * (Nx // 2) + [1] * (Nx - Nx // 2) + [1] * (Nx // 2) + [0] * (Nx - Nx // 2) # polarised

    # variation_multipliers = [-0.1, 0, 0.5, 0.75, 1.5, 2, 10] # 1x is assumed
    # variation_multipliers = [0.1, 0.25, 0.5, 2, 5, 10] # 1x is assumed
    variation_multipliers = [0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3]  # 1x is assumed
    index_for_100x = 4
    xticks = [format(f"{x * 100:.2f}%") for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")

    tasks = generate_tasks(MODELS.GOEHRING, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL, v_func=v_func_zero), {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity maintenance")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    all_results_by_variable = run_grouped_tasks(tasks)
    baseline, results_by_variable = split_baseline_from_results(MODELS.GOEHRING, all_results_by_variable, index_for_100x)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL), x_axis_labels=xticks)


if __name__ == '__main__':
    test_polarity_establishment(Nx=40)
    # test_polarity_maintenance(Nx=40)
    plt.show()


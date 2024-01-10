import numpy as np
from matplotlib import pyplot as plt
from .. import model_task_handler
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


Nx = 100

DEFAULT_PARAMETERS = {
    "key_varied": "",
    "label": "goehring",
    "points_per_second": 0.1,

    # General Setup Variables
    "Nx": Nx,  # number of length steps
    "L": 134.6,  # length of region
    "x0": 0,
    "xL": 67.3,  # L / 2
    "t0": 0,
    "tL": 3600,

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
    "v_func": default_v_func,
}

# INITIAL_CONDITION = [0.5] * Nx + [0] * Nx
INITIAL_CONDITION = [2.828] * Nx + [0.025] * Nx  # the values we get when running baseline with v=0
INITIAL_CONDITION_POLARISED = [1]*(Nx//2) + [0]*(Nx-Nx//2) + [0]*(Nx//2) + [1]*(Nx-Nx//2)


variation_multipliers = [-0.1, 0, 0.5, 0.75, 1.5, 2, 10] # 1x is assumed
index_for_100x = 4


def generate_tasks(other_param_vals: dict) -> list[list[tuple]]:
    sort_i = 1

    default_params = {**DEFAULT_PARAMETERS, **other_param_vals}
    tasks: list[list(tuple)] = [[(MODELS.GOEHRING, {**default_params, "sort": sort_i})]]  # we add baseline task
    sort_i += 1

    for key in DEFAULT_PARAMETERS.keys():
        task_list = []

        if key not in ["key_varied", "label", "points_per_second", "Nx", "L", "x0", "xL", "t0", "tL", "v_func"]:  # exclude run conditions
            for multiplier in variation_multipliers:
                task_list.append((MODELS.GOEHRING, {**default_params,
                    "label": format(f"{key}={default_params[key] * multiplier:.2f}"),
                    "key_varied": format(f"{key}"),
                    key: default_params[key] * multiplier, "sort": sort_i}))
            
            sort_i += len(variation_multipliers)
            tasks.append(task_list)

    return tasks

def test_polarity_establishment():
    tasks = generate_tasks({"label": ".", "initial_condition": INITIAL_CONDITION})

    print(f"Testing polarity establishment")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    baseline_result = None
    results_by_variable = []

    for task_set_i in range(0, len(tasks)):
        print(f"Running Task Set {task_set_i}")

        task_set = tasks[task_set_i]

        results = model_task_handler.run_tasks_parallel(task_set)

        results.sort(key=lambda res: 0 if res[1] == "FAILURE" or not "sort" in res[2] else res[2]["sort"])  # order when plotting multiple solutions on single figure

        sol_list = []
        kvals_list = []
        has_failure = False

        for res in results:
            if res[1] == "FAILURE":
                has_failure = True
            #     continue
            
            if task_set_i == 0 and len(task_set) == 1:
                baseline_result = (res[1], res[2])
                model_module.animate_plot(res[1], res[2], rescale=True)
            
            sol_list.append(res[1])
            kvals_list.append(res[2])

        assert len(sol_list) == len(kvals_list)

        if len(sol_list) > 1: # I exclude baseline (only single-sol) run for now
            # if not has_failure:
            #     model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_P=False, label="goehring_sensitivity_establishment_comparison")
                
            assert baseline_result is not None
            sol_list.insert(index_for_100x, baseline_result[0])
            kvals_list.insert(index_for_100x, baseline_result[1])
            results_by_variable.append((sol_list, kvals_list))


    assert baseline_result is not None
    assert len(results_by_variable) == len(tasks) - 1

    xticks = [format(f"{x*100}%") for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity", x_axis_labels=xticks)
    # model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_P=False)

    plt.show()


def v_func_zero(kvals, x, t):
    return 0


def test_polarity_maintenance():
    tasks = generate_tasks({"label": ".", "initial_condition": INITIAL_CONDITION_POLARISED, "v_func": v_func_zero})

    print(f"Testing polarity maintenance with v=0")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0, len(tasks))])} tasks")

    baseline_result = None
    results_by_variable = []

    for task_set_i in range(0, len(tasks)):
        print(f"Running Task Set {task_set_i}")

        task_set = tasks[task_set_i]

        results = model_task_handler.run_tasks_parallel(task_set)

        results.sort(key=lambda res: 0 if res[1] == "FAILURE" or not "sort" in res[2] else res[2][
            "sort"])  # order when plotting multiple solutions on single figure

        sol_list = []
        kvals_list = []
        has_failure = False

        for res in results:
            if res[1] == "FAILURE":
                has_failure = True
            #     continue

            if task_set_i == 0 and len(task_set) == 1:
                baseline_result = (res[1], res[2])

            sol_list.append(res[1])
            kvals_list.append(res[2])

        assert len(sol_list) == len(kvals_list)

        if len(sol_list) > 1:  # I exclude baseline (only single-sol) run for now
            if not has_failure:
                model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_P=False,
                                                       label="goehring_sensitivity_maintenance_comparison")

            assert baseline_result is not None
            sol_list.insert(index_for_100x, baseline_result[0])
            kvals_list.insert(index_for_100x, baseline_result[1])
            results_by_variable.append((sol_list, kvals_list))

    assert baseline_result is not None
    assert len(results_by_variable) == len(tasks) - 1

    xticks = [format(f"{x * 100}%") for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity", x_axis_labels=xticks)
    # model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_P=False)

    plt.show()


if __name__ == '__main__':
    test_polarity_establishment()
    # test_polarity_maintenance()
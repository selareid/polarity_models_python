from matplotlib import pyplot as plt
from .. import model_task_handler
from ..models import MODELS, model_to_module

model_module = model_to_module(MODELS.TOSTEVIN)
Nx = 100

DEFAULT_PARAMETERS = {
    "key_varied": "",
    "label": "tostevin",
    "points_per_second": 0.1,

    # General Setup Variables
    "Nx": Nx,  # number of length steps
    "L": 50.0,  # length of region - model parameter
    "x0": 0,
    "xL": 50,  # L
    "t0": 0,
    "tL": 3600,

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

INITIAL_CONDITION = [1]*Nx + [0]*Nx + [0]*Nx + [1]*Nx \
         + [50] # L
INITIAL_CONDITION_POLARISED = [1]*(Nx//2) + [0]*(Nx-Nx//2) + [0.5]*Nx \
         + [0]*(Nx//2) + [1]*(Nx-Nx//2) + [0.5]*Nx + [50] # L


variation_multipliers = [-0.1, 0, 0.5, 0.75, 1.5, 2, 10] # 1x is assumed
# variation_multipliers = [0.5, 0.75, 1.5, 2] # 1x is assumed
index_for_100x = 4

def generate_tasks(other_param_vals: dict):
    sort_i = 1

    default_params = {**DEFAULT_PARAMETERS, **other_param_vals}
    tasks = [[(MODELS.TOSTEVIN, {**default_params, "sort": sort_i})]]
    sort_i += 1

    for key in DEFAULT_PARAMETERS.keys():
        task_list = []

        if key not in ["key_varied", "label", "points_per_second", "Nx", "L", "x0", "xL", "t0", "tL", "a_func"]:  # exclude run conditions
            for multiplier in variation_multipliers:
                task_list.append((MODELS.TOSTEVIN, {**default_params,
                    "label": format(f"{key}={default_params[key] * multiplier:.2f}"),
                    "key_varied": format(f"{key}"),
                    key: default_params[key] * multiplier}))
            
            sort_i += len(variation_multipliers)
            tasks.append(task_list)

    return tasks

def test_polarity_establishment():
    tasks = generate_tasks({"label": "", "initial_condition": INITIAL_CONDITION})

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
                # continue

            if task_set_i == 0 and len(task_set) == 1:
                baseline_result = (res[1], res[2])
            
            sol_list.append(res[1])
            kvals_list.append(res[2])

        assert len(sol_list) == len(kvals_list)

        if len(sol_list) > 1: # I exclude baseline (only single-sol) run for now
            # if not has_failure:
            #     # can do some multi plotting or smthng

            assert baseline_result is not None
            sol_list.insert(index_for_100x, baseline_result[0])
            kvals_list.insert(index_for_100x, baseline_result[1])
            results_by_variable.append((sol_list, kvals_list))            


    assert baseline_result is not None

    xticks = [format(f"{x*100}%") for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")
    model_module.plot_variation_sets(results_by_variable, label="Tostevin Polarity", x_axis_labels=xticks)
    # model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Ac=False,plot_Pm=False,plot_Pc=False)

    plt.show()


def a_func_zero(kvals, lt, x):
    return 0


def test_polarity_maintenance():
    tasks = generate_tasks({"label": "", "initial_condition": INITIAL_CONDITION_POLARISED, "a_func": a_func_zero})

    print(f"Testing polarity maintenance with a=0")
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
                # continue

            if task_set_i == 0 and len(task_set) == 1:
                baseline_result = (res[1], res[2])

            sol_list.append(res[1])
            kvals_list.append(res[2])

        assert len(sol_list) == len(kvals_list)

        if len(sol_list) > 1:  # I exclude baseline (only single-sol) run for now
            # if not has_failure:
            #     # can do some multi plotting or smthng

            assert baseline_result is not None
            sol_list.insert(index_for_100x, baseline_result[0])
            kvals_list.insert(index_for_100x, baseline_result[1])
            results_by_variable.append((sol_list, kvals_list))

    assert baseline_result is not None

    xticks = [format(f"{x * 100}%") for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")
    model_module.plot_variation_sets(results_by_variable, label="Tostevin Polarity", x_axis_labels=xticks)

    plt.show()



if __name__ == '__main__':
    test_polarity_establishment()
    # test_polarity_maintenance()

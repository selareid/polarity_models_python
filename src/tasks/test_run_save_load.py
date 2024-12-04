# Basic run of goehring variation tasks / sensitivity analysis 
# using fewer nodes/xticks and parameters in order to test
# saving and loading of runs

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from matplotlib import pyplot as plt
from src.tasks import variation_task_helper as t_helper
from ..models import MODELS, model_to_module

MODULE_GOEHRING = model_to_module(MODELS.GOEHRING)

test_parameters = {
    "label": "goehring load/save test",
    "D_A": 0.28, "k_onA": 8.58 * 10**(-3)
    }

if __name__ == '__main__':
    Nx = 50
    tL = 3000
    filename = "test_run_save_load_save_0123"
    
    initial_condition = [2.828] * Nx + [0.025] * Nx

    variation_multipliers, index_for_100x = t_helper.get_variation_multiplier("sparse")
    xticks = t_helper.get_xticks(variation_multipliers, index_for_100x)

    tasks = t_helper.generate_tasks(MODELS.GOEHRING, variation_multipliers,
        test_parameters, {"Nx": Nx, "tL": tL, "initial_condition": initial_condition})

    print("Testing save/load using goehring model")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    print("Attempting to load")
    load_data = t_helper.load_runs(filename)
    if load_data[0]:
        print("Loading succeeded")
        baseline = load_data[1]
        results_by_variable = load_data[2]
    else:
        print("Loading failed")
        baseline, results_by_variable = t_helper.split_baseline_from_results(MODELS.GOEHRING,
            t_helper.run_grouped_tasks(tasks), index_for_100x)

        print("Saving run data")
        t_helper.save_runs(filename, tasks, baseline, results_by_variable)
        print("Saving finished")

    print("Plotting")
    MODULE_GOEHRING.plot_variation_sets(results_by_variable, label="testing testing",
        x_axis_labels=xticks)

    print("Finished")
    plt.show(block=True)


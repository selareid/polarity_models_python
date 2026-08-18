# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '12'
import numpy as np
from matplotlib import pyplot as plt
import pickle
from dataclasses import asdict, fields
from pathlib import Path
from time import time

from polarity.model_enums import MODELS, model_to_module
from polarity.utilities import model_task_handler, figure_helper as fh


model = MODELS.PAR3ND
# model = MODELS.GOEHRING
multipliers = np.round(np.linspace(0.5, 1.5, num = 41), decimals=5) 

tL = 300*60
store_times = np.linspace(0, tL, num = 121)
n_procs = 12


def run_parameter_sweep(param_name, default_value, other_args = {}):
    # Generate the parameter values we want to run
    # assert (np.min(multipliers) > 0.0) # Otherwise the sigfigs calc will crash
    new_param_vals = [m*default_value for m in multipliers]

    # Loop over new values and generate a list of tasks
    task_list = []
    for mult, param_val in zip(multipliers, new_param_vals):
        # Add to the task list
        label = f"{param_name}_mult={mult}"
        task = (model, {**other_args, param_name: param_val, "t_eval": store_times, "tL": tL, 
                        "calc_ss_initial_condition": True, "label": label})
        task_list.append(task)

    # Run tasks in parallel
    res_list = model_task_handler.run_tasks_parallel(task_list, NUMBER_OF_PROCESSES=n_procs)

    # Convert the setup to a dictionary, so it's easier to work with later
    res_list_conv = {label: (res[0], asdict(res[1])) for label, res in res_list}

    # Add in the scaling factor for the parameter for reference later
    for label, res in res_list_conv.items():
        label_split = label.split("=",1)
        assert(len(label_split) == 2)
        res[1][label_split[0]] = float(label_split[1])

    # Return as a list of results ordered by increasing multiplier
    return([res_list_conv[sorted_label] for sorted_label in sorted(res_list_conv.keys())])


if __name__ == '__main__':
    
    # Get the parameter names from the model's DEFAULT_PARAMETERS dataclass, just consider rate parameters and densities
    default_parameters = model_to_module(model).DEFAULT_PARAMETERS
    parameters_list = [f.name for f in fields(default_parameters) if f.name.startswith(('k', 'rho'))]
    print(parameters_list)
    
    output_dir = model_task_handler.DATA_DIR / f"sensitivity_establishment_{model.name}"
    if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Run for each parameter and store
    start_time = time()
    for param_name in parameters_list:
        print(f"Running establishment sensitivity analysis for parameter {param_name}")
        default_value = getattr(default_parameters, param_name)
        res_list = run_parameter_sweep(param_name, default_value)  

        # Save the full dataset
        output_filename = Path(output_dir, f"{param_name}_sweep.pkl")
        with open(output_filename, 'wb') as f:
                pickle.dump(res_list, f)
    
    print(f"Total time taken for sensitivity analysis: {(time()-start_time)/60:.1f} minutes")

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
model_module = model_to_module(model)

multipliers = np.round(np.linspace(0.5, 1.5, num = 41), decimals=5) 
tL = 300*60
store_times = np.linspace(0, tL, num = 121)
n_procs = 12

def main():
    # Establishment sensitivity
    output_dir_est = model_task_handler.DATA_DIR / f"sensitivity_establishment_{model.name}"
    if not Path(output_dir_est).exists():
        Path(output_dir_est).mkdir(parents=True, exist_ok=True)
    # run_establishment_sensitivity_analysis(output_dir_est)
    
    # Loss (mid-secretory phase) sensitivity
    output_dir_MS = model_task_handler.DATA_DIR / f"sensitivity_MS_loss_{model.name}"
    if not Path(output_dir_MS).exists():
        Path(output_dir_MS).mkdir(parents=True, exist_ok=True)
    # run_loss_sensitivity_analysis(output_dir_MS, rho_A_mult=0.5, rho_P_mult=0.4)

    # Loss (late-secretory phase) sensitivity
    output_dir_LS = model_task_handler.DATA_DIR / f"sensitivity_LS_loss_{model.name}"
    if not Path(output_dir_LS).exists():
        Path(output_dir_LS).mkdir(parents=True, exist_ok=True)
    run_loss_sensitivity_analysis(output_dir_LS, rho_A_mult=0.5, rho_P_mult=0.15)


def run_parameter_sweep(param_name, default_value, param_multipliers, update_args = {}) -> list[tuple[str, tuple]]:
    # Generate the parameter values we want to run
    # assert (np.min(multipliers) > 0.0) # Otherwise the sigfigs calc will crash
    new_param_vals = [m*default_value for m in param_multipliers]

    # Loop over new values and generate a list of tasks
    task_list = []
    for mult, param_val in zip(param_multipliers, new_param_vals):
        # Add to the task list
        label = f"{param_name}_mult={mult}"
        task = (model, {
                    **update_args, 
                    "label": label, 
                    param_name: param_val,
                    })
        task_list.append(task)

    # Run tasks in parallel
    res_list = model_task_handler.run_tasks_parallel(task_list, NUMBER_OF_PROCESSES=n_procs)

    # Add in the scaling factor for the parameter for reference later
    for label, res in res_list:
        label_split = label.split("=",1)
        assert(len(label_split) == 2)
        res[1][label_split[0]] = float(label_split[1])

    return([res for _, res in res_list]) # Don't need the label (included in the setup dict)



def run_establishment_sensitivity_analysis(output_dir):
    # Get the parameter names from the model's DEFAULT_PARAMETERS dataclass, just consider rate parameters and densities
    default_parameters = model_to_module(model).DEFAULT_PARAMETERS
    parameters_list = [f.name for f in fields(default_parameters) if f.name.startswith(('k', 'rho'))]
    print(parameters_list)

    # Run for each parameter and store
    start_time = time()
    for param_name in parameters_list:
        print(f"Running establishment sensitivity analysis for parameter {param_name}")
        default_value = getattr(default_parameters, param_name)
        update_args = {
             "t_eval": store_times, "tL": tL, 
             "calc_ss_initial_condition": True
            }
        res_list = run_parameter_sweep(param_name, default_value, multipliers, update_args)

        # Save the full dataset
        output_filename = Path(output_dir, f"{param_name}_sweep.pkl")
        with open(output_filename, 'wb') as f:
                pickle.dump(res_list, f)
    
    print(f"Total time taken for establishment sensitivity analysis: {(time()-start_time)/60:.1f} minutes")


def run_loss_sensitivity_analysis(output_dir, rho_A_mult, rho_P_mult):
    default_parameters = model_to_module(model).DEFAULT_PARAMETERS

    # Mid-secretory phase values for rho_A and rho_P
    mid_sec_params = {
        "rho_A": rho_A_mult*default_parameters.rho_A,
        "rho_P": rho_P_mult*default_parameters.rho_P
    }

    # List of parameters to sweep over for loss sensitivity analysis
    parameters_list = [f.name for f in fields(default_parameters) 
                       if f.name.startswith(('k', 'rho')) and f.name not in ['rho_A', 'rho_P']]
    print(parameters_list)

    # Get the initial conditions (proliferative phase polarised state)
    polarised_ic = model_module.run_for_polarised_initial_condition()

    # Run for each parameter and store
    start_time = time()
    for param_name in parameters_list:
        print(f"Running loss sensitivity analysis for parameter {param_name}")
        default_value = getattr(default_parameters, param_name)
        update_args = {
             **mid_sec_params,
             "t_eval": store_times, "tL": tL, 
             "initial_condition": polarised_ic
            }
        res_list = run_parameter_sweep(param_name, default_value, multipliers, update_args)

        # Save the full dataset
        output_filename = Path(output_dir, f"{param_name}_sweep.pkl")
        with open(output_filename, 'wb') as f:
                pickle.dump(res_list, f)
    
    print(f"Total time taken for loss sensitivity analysis: {(time()-start_time)/60:.1f} minutes")


if __name__ == '__main__':
    main()

import numpy as np
import pickle
from dataclasses import asdict
from pathlib import Path
from time import time

from polarity.model_enums import MODELS, model_to_module
from polarity.utilities import figure_helper as fh
from run_parameter_sweep import run_parameter_sweep

# Global parameters
model = MODELS.PAR3ND
# model = MODELS.GOEHRING
model_module = model_to_module(model)
data_dir = fh.DATA_DIR / f"cycle_sweeps_{model.name}"

multipliers = np.round(np.linspace(0.1, 1.5, 57), decimals=3)
tL = 300*60
store_times = np.linspace(0, tL, num = 121)


if __name__ == '__main__':
    start_time = time()

    # Output directory
    if not Path(data_dir).exists():
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    # Parameters to sweep over
    param_i = "rho_A"
    param_j = "rho_P"

    # Initial condition
    polarised_ic = model_module.run_for_polarised_initial_condition()
    
    # Get the default values for the parameters
    default_value_i = getattr(model_module.DEFAULT_PARAMETERS, param_i)
    default_value_j = getattr(model_module.DEFAULT_PARAMETERS, param_j)

    # We run and store for each level of param_i, sweeping over param_j
    for level, mult in enumerate(multipliers):
        param_i_val = mult*default_value_i
        print(f"Running cycle sensitivity for {param_i} multiplier {mult}")
        update_args = {
            param_i: param_i_val,
            "initial_condition": polarised_ic,
            "t_eval": store_times, "tL": tL
        }
        res_list = run_parameter_sweep(param_j, default_value_j, multipliers, update_args)
        # Add the param_i multiplier to the results for reference later
        for sol, setup in res_list:
            setup[f"{param_i}_mult"] = mult
        # Save
        output_filename = Path(data_dir, f"param_i_level_{level}.pkl")
        with open(output_filename, 'wb') as f:
            pickle.dump(res_list, f)
    # Approx. 2 hours for 57 levels of each parameter on 12 cores, Mac Studio
    print(f"Total time taken for cycle sensitivity: {(time()-start_time)/60:.1f} minutes")
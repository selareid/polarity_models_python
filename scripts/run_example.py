# Example for running a model and changing parameters

# makes running many tasks quicker when there are a lot of them
# # https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

from polarity.utilities import figure_helper as fh
from polarity.model_enums import MODELS, model_to_module

# Choose which model to run
model = MODELS.PAR3ND
# model = MODELS.GOEHRING
# model = MODELS.PAR3ADD

# choose parameters for model simulation. list of default
# parameters in src.models.par3addition DEFAULT_PARAMETERS constant
parameters = {
                "points_per_second": 0.1, # Can be used instead of t_eval
                # "t_eval": [0, 3, 100],  # can use instead of points_per_second
                # change rho_A and kPA from default
                "rho_A": 1.2,
                "kPA": 1.8,
                "tL": 1500  # end timestep
                }

# Run model
result, setup = model_to_module(model).run_model(parameters)
# could also use model_task_handler.load_or_run(identifier, tasks)
# or model_task_handler.run_tasks(tasks)

# results.y[<spatial point>, <time point>] gives integration output
# the time index will correspond to t_eval if used
# otherwise will be determined by points_per_second

# setup gives the parameters used

filename = model.name + "_example_animation"
fh.animate_plot(result, setup, save_file=filename)



# Example for running par3add model and changing parameters

# makes running many tasks quicker when there are a lot of them
# # https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

from polarity.utilities import model_task_handler
from polarity.model_enums import MODELS, model_to_module

model_module = model_to_module(MODELS.PAR3ADD)


if __name__ == '__main__':
    # choose parameters for model simulation. list of default
    # parameters in src.models.par3addition DEFAULT_PARAMETERS constant
    parameters = {
                    "points_per_second": 0.1,
                    # "t_eval": [0, 3, 1000],  # can use instead of points_per_second
                    "tL": 1000,  # end timestep
                    "Nx": 100,  # spatial steps
                    "initial_condition": [1]*100 + [1]*100 + [1]*100 + [0]*100,

                    # change rho_J and kPA from default
                    "rho_J": 1,
                    "kPA": 1.8,
                    }

    # array of tasks
    tasks = [(MODELS.PAR3ADD, parameters)]

    results = model_task_handler.run_tasks_parallel(tasks)
    # could also use model_task_handler.load_or_run(identifier, tasks)
    # or model_task_handler.run_tasks(tasks)

    for res in results:
        # check for integration failure
        if res[1] == "FAILURE":
            continue

        model_module.animate_plot(res[1], res[2], save_file=True)

        # res[1].y[<spatial point>, <time point>] gives integration output
        # the time index will correspond to t_eval if used
        # otherwise will be determined by points_per_second

        # res[2] gives the parameters used

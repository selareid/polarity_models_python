# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src.tasks import variation_task_helper
import matplotlib
import numpy as np
from src.models.metric_functions import polarity_get_all
from matplotlib import pyplot as plt
from ..models import MODELS, model_to_module


model_module = model_to_module(MODELS.PAR3ADD)


# line between (x1,y1) and (x2,y2)
# input x, get y
# def straight_line_transform(x1, y1, x2, y2, x):
#     y = (y2 - y1)*(x - x1)/(x2 - x1) + y2
#     return y


# def v_func_hopefully_forces_polarity_more_strongly_than_default(kvals, x, t):
#     v_time = 1500  # 600
#     time_factor = 1 / np.maximum(1, t / 10 - v_time / 10)

#     center = 15  # kvals["xL"] / 4
#     sd = np.minimum(center / 4, (kvals["xL"] - center) / 4)
#     peak = 0.1

#     return time_factor * peak * np.exp(-(x - center) ** 2 / (2 * sd ** 2))


def v_func_def(kvals, x, t):
    v_time = 600
    time_factor = 1 / np.maximum(1, t / 10 - v_time / 10)

    center = kvals["xL"] / 4
    sd = np.minimum(center / 4, (kvals["xL"] - center) / 4)
    peak = 0.1

    return time_factor * peak * np.exp(-(x - center) ** 2 / (2 * sd ** 2))


def v_func_zero(kvals, x, t):
    return 0


def get_default_parameters(Nx, tL, v_func):  # , v_func
    return {
        "label": "par3addition",
        "points_per_second": 0.01,

        # General Setup Variables
        "Nx": Nx,  # number of length steps
        "L": 134.6,  # length of region
        "x0": 0,
        "xL": 67.3,  # L / 2
        "t0": 0,
        "tL": tL,

        # Model parameters and functions #
        "psi": 0.174,  # surface to volume conversion factor

        # diffusion coefficients
        "D_J": 0.28,
        "D_M": 0.28,
        "D_A": 0.28,
        "D_P": 0.15,

        # velocity coefficients
        "sigmaJ": 1,
        "sigmaM": 1,
        "sigmaP": 1,

        # aPar complex change coefficients
        "k1": 8.58 * 10**(-3),
        "k2": 6.5 * 10**(-3),

        # antagonism
        "kJP": 0.190,
        "kMP": 0.190,
        "kAP": 0.190,
        "kPA": 2.0,

        # antagonism exponents
        "alpha": 1,
        "beta": 2,

        # total amount in system
        "rho_A": 1.56,
        "rho_J": 1.56,
        "rho_P": 1.0,

        "konJ": 8.58 * 10**(-3),
        "konA": 0,
        "konP": 4.74 * 10**(-2),

        "koffJ": 7.3 * 10**(-3),
        "koffA": 5.4 * 10**(-3),
        "koffP": 7.3 * 10**(-3),

        "v_func": v_func,

        # these two added later
        "kMP": 0,
        "kAP": 0,
    }


# range_info is of form (start, stop, steps)
# gives tasks for two variables varied at a time
# number of tasks output is steps1*steps2
def get_parameter_space_tasks(def_params, first_key, first_range, second_key, second_range):
    task_list = []

    for v1 in np.linspace(first_range[0], first_range[1], first_range[2]):
        for v2 in np.linspace(second_range[0], second_range[1], second_range[2]):
            parameters = {**def_params, first_key: v1, second_key: v2,
                              "keys_varied": [first_key, second_key],
                              "sort": 1000*v1 + v2,
                              "label": f"par3Add,{first_key}:{v1:.4f},{second_key}:{v2:.4f}"
                          }

            task_list.append(
                (MODELS.PAR3ADD, parameters)
            )

    return task_list


# generate a string for each parameter variation
# string consists of just the chosen parameter values
# uses param_list for a consistent ordering
# param_list - ordered list of variables
# kvals - list of the parameter values
# def get_variation_key(param_list, kvals):
#     key = ""
#     for param in param_list:
#         key = key + format(f"{kvals[param]:.5f},")
#     return key


# tasks_set: {key: [<tasks>]}, return same but results instead of tasks
def run_all_task_sets(tasks_set: dict):
    flat_tasks_set = [set for set in tasks_set.values()]

    results_set_list = variation_task_helper.run_grouped_tasks(flat_tasks_set)

    results_set = {}

    for result_list in results_set_list:
        keys_varied: list = result_list[0][2]["keys_varied"]
        set_key = keys_varied[0]+keys_varied[1]
        results_set[set_key] = []

        for result in result_list:
            assert result[2]["keys_varied"] == keys_varied
            results_set[set_key].append(result)

    return results_set


def plot_two_variable_variation(result_set: list[tuple]):
    # make scatter plot
    # colour for m
    plt.figure()

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["grey", "yellow", "violet", "blue"])
    norm = plt.Normalize(0, 1)

    key1, key2 = result_set[0][2]["keys_varied"]
    Nx = result_set[0][2]["Nx"]

    for res in result_set:
        sol = res[1]
        kvals = res[2]

        assert kvals["keys_varied"][0] == key1 and kvals["keys_varied"][1] == key2

        v1, v2 = kvals[key1], kvals[key2]

        # check failure
        if sol == "FAILURE":
            plt.scatter(v1, v2, marker="x", s=100)
            continue

        p, _, marker = polarity_get_all(kvals["X"], sol.y[2*Nx:3*Nx, -1], sol.y[3*Nx:, -1], Nx)
        alpha = np.minimum(1, p + 0.2)
        plt.scatter(v1, v2, c=p, cmap=cmap, norm=norm, alpha=alpha, marker=marker, s=100)

    clb = plt.colorbar()
    clb.ax.set_title("polarity")

    plt.xlabel(key1)
    plt.ylabel(key2)
    plt.title("par3Add Variation Nx=" + str(Nx) + ",tL=" + str(tL))


if __name__ == '__main__':
    Nx = 30  #100
    tL = 6000

    default_params = {**get_default_parameters(Nx, tL, v_func=v_func_def),
                        # while testing, put chosen parameter changes here
                        "k2": 0.0075,
                        "kJP": 0.0037,
                        "kMP": 0.006,
                        "kPA": 0.1196,
                        "rho_A": 1.323,
                        "rho_J": 1.125,
                        "koffJ": 0.0001,

                        "rho_A": 1.323,
                        # "initial_condition": [1]*(Nx*3)+[0]*Nx,
                        # "v_func": v_func_zero
                        "kAP": 0.19,  # 0,
                        }

    key_params: list[str] = ["rho_A", "kAP"]  # what parameters to consider (can be more than two)

    # range and number of variations
    variation_max_multiplier = 1.25
    variation_min_multiplier = 0.75
    variation_steps = 20

    # This gives the animated plot for the aPar feedback loop section of report
    if False:
        default_params = {**default_params, 
                        "k2": 0.05,
                        "kJP": 0.1,
                        "kMP": 0.006,
                        "kAP": 0,
                        "kPA": 0.1196,
                        "rho_A": 1.32342534,
                        "rho_J": 1.125,
                        "rho_P": 1.0904,
                        "konJ": 0.08,
                        "koffJ": 0.0001,
                        }

        # if you use following three lines, you get animated plot
        variation_max_multiplier = 1
        variation_min_multiplier = 1
        variation_steps = 1

        key_params = ["kAP", "k2"]  # overriding this just so we only get 1 animated figure out


    tasks_set = {}

    # generate tasks, varying two parameters for each set
    for key1_i in range(len(key_params)):
        key1 = key_params[key1_i]
        for key2_i in range(key1_i+1, len(key_params)):
            key2 = key_params[key2_i]
            tasks_set[key1+key2] = (get_parameter_space_tasks(default_params,
                                             key1,
                                       (default_params[key1]*variation_min_multiplier, default_params[key1]*variation_max_multiplier, variation_steps),
                                             key2,
                                       (default_params[key2]*variation_min_multiplier, default_params[key2]*variation_max_multiplier, variation_steps))
                                    )
        # break  # uncomment if you only want the plots with key_param[0] on the bottom axis
    results_by_set: dict = run_all_task_sets(tasks_set)


    # plot each set
    # then (manually?) see points where polarisation is achieved
    for results_set in results_by_set.values():
        if variation_steps > 1:
            plot_two_variable_variation(results_set)

        for res in results_set:
            if res[1] != "FAILURE":
                sol = res[1]
                kvals = res[2]

                if polarity_get_all(kvals["X"], sol.y[2*Nx:3*Nx, -1], sol.y[3*Nx:, -1], Nx)[0] >= 0.75:
                    print(f"POLARISED GOOD AT {kvals['keys_varied']},{kvals[kvals['keys_varied'][0]]},"
                          + f"{kvals[kvals['keys_varied'][1]]}"
                          + f' P:{polarity_get_all(kvals["X"], sol.y[2*Nx:3*Nx, -1], sol.y[3*Nx:, -1], Nx)[0]}')
                    # # BEWARE! uncommenting below has the potential to open hundreds of plots
                    # model_module.plot_final_timestep(res[1], res[2])
                    # model_module.animate_plot(res[1], res[2], save_file=True)
                    

                if variation_steps == 1:
                    model_module.animate_plot(res[1], res[2], save_file=True)

    plt.show()
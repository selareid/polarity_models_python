# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from src import model_task_handler
from src.models.goehring import animate_plot, plot_final_timestep
from src.models.metric_functions import polarity_get_all
from ..models import MODELS, model_to_module

model_module = model_to_module(MODELS.GOEHRING)


def new_v_func(kvals, x, t):
    time_factor = 1 / np.maximum(1, t/10-kvals["v_time"]/10)

    center = kvals["v_position"]  # kvals["xL"] / 4
    sd = np.minimum(center/4, (kvals["xL"]-center)/4)
    peak = 0.1

    return time_factor * peak*np.exp(-(x-center)**2 / (2 * sd**2))


def get_default_parameters(Nx, tL, v_func=new_v_func):
    return {"label": "goehring",
            "points_per_second": 0.01,

            # General Setup Variables
            "Nx": Nx,  # number of length steps
            "L": 134.6,  # length of region
            "x0": 0,
            "xL": 67.3,  # L / 2
            "t0": 0,
            "tL": tL,

            "v_func": v_func,

            # specifically for this v-func variation
            "v_time": 600,
            "v_position": 67.3 / 4,
            }


def run_variations(Nx, tL, initial_condition):
    # [[time] space]
    X = np.linspace(0, 67.3, 20 + 2)  # x0 to xL
    # X = np.linspace(0, 67.3, 10 + 2)  # x0 to xL
    X = X[1:-1]  # exclude start and end of space (want v=0 at endpoints)
    T = np.linspace(30, 3000, 20)
    # T = np.linspace(30, 2000, 20)

    tasks_list = []

    for t in T:
        for x in X:
            tasks_list.append((MODELS.GOEHRING, {**get_default_parameters(Nx, tL),
                                                 "initial_condition": initial_condition,
                                                 "label": format(f"x={x},t={t}"),
                                                 "v_time": t,
                                                 "v_position": x, }))

    # run tasks
    results = model_task_handler.run_tasks_parallel(tasks_list)

    # make scatter plot
    # colour for m
    plt.figure()
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["grey", "yellow", "violet", "blue"])
    norm = plt.Normalize(0, 1)

    for res in results:
        sol = res[1]
        kvals = res[2]

        # check failure
        if sol == "FAILURE":
            plt.scatter(kvals["v_position"], kvals["v_time"], marker="x", s=100)
            continue

        p, _, marker = polarity_get_all(kvals["X"], sol.y[:kvals["Nx"], -1], sol.y[kvals["Nx"]:, -1], kvals["Nx"])
        alpha = np.minimum(1, p + 0.2)
        plt.scatter(kvals["v_position"], kvals["v_time"], c=p, cmap=cmap, norm=norm, alpha=alpha, marker=marker, s=100)

    clb = plt.colorbar()
    clb.ax.set_title("polarity")

    plt.xlabel("peak position")
    plt.ylabel("time before dropping")
    plt.title("goehring v_function variation Nx=" + str(Nx) + ",tL=" + str(tL))


def run_interesting(Nx, tL, initial_condition, v_x, v_t):
    print(f"Running Interesting Case x:{v_x},t:{v_t}")

    _, sol, kvals = model_task_handler.run_tasks([(MODELS.GOEHRING, {**get_default_parameters(Nx, tL),
                                                 "initial_condition": initial_condition,
                                                 "label": format(f"x={v_x},t={v_t}"),
                                                 "v_time": v_t,
                                                 "v_position": v_x, })])[0]
    animate_plot(sol, kvals)
    plot_final_timestep(sol, kvals)


if __name__ == '__main__':
    Nx = 100
    tL = 3000

    initial_condition = [2.828] * Nx + [0.025] * Nx  # the values we get when running baseline with v=0


    run_variations(Nx, tL, initial_condition)
    # run_interesting(Nx, tL, initial_condition, 25, 1600)

    plt.show()


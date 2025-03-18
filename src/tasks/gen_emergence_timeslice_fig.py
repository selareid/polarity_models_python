# generate illustration of par3add (new model) over time

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import matplotlib
from matplotlib import pyplot as plt
from src import model_task_handler
from ..models import MODELS, model_to_module, metric_functions
from src import figure_helper


def v_func_zero(kvals, x, t):
    return 0

Module_Par3Add = model_to_module(MODELS.PAR3ADD)

params = {
    "psi": 0.174,

    "D_J": 0.28,
    "D_M": 7.5*10**(-2),
    "D_A": 0.28,
    "D_P": 0.15,

    "k1": 9.01*10**(-3),
    "k2": 1.64*10**(-3),

    "kJP": 6.16*10**(-2),
    "kMP": 4.41*10**(-2),
    "kAP": 4.61*10**(-1),
    "kPA": 2,

    "rho_J": 1.2,
    "rho_A": 1.56,
    "rho_P": 1,

    "konJ": 1.4*10**(-2),
    "konP": 4.74*10**(-2),

    "koffJ": 1.17*10**(-3),
    "koffM": 8.44*10**(-3),
    "koffA": 2.65*10**(-3),
    "koffP": 7.3*10**(-3),

    # not used in writeup
    "konA": 0,  
    "sigmaJ": 1, "sigmaM": 1, "sigmaP": 1,
    "alpha": 1, "beta": 2,
   }


def main():
    Nx = 100
    tL = 9000
    # points_per_second = 0.1
    plot_times = [0, 300, 800, 9000]
    
    # run to reach apically-dominant steady-state
    start_initial_condition = [1] * (Nx*3) + [0]*Nx
    task_hom = [(MODELS.PAR3ADD, {**params,
        "Nx": Nx,
        "tL": tL, "v_func": v_func_zero,
        "initial_condition": start_initial_condition,
        "label": "par3add_get_a_dominant_ss"
        })]
    res_hom = model_task_handler.load_or_run("par3add_get_dominant_ss", task_hom)[0]
    initial_condition = res_hom[1].y[:,-1]

    # run to polarise
    task_emergence = [(MODELS.PAR3ADD, {**params,
        "Nx": Nx,
        "tL": tL,
        "t_eval": plot_times,
        "initial_condition": initial_condition,
        "label": "par3add_emergence"
        })]
    res_emg = model_task_handler.load_or_run("par3add_emergence", task_emergence)[0]

    # plot
    fig, axs = plt.subplots(nrows=1, ncols=len(plot_times), sharex=True, sharey=True)

    for i in range(0, len(plot_times)):
        # plot subfigure
        assert res_emg[1].t[i] == plot_times[i]
        ax = axs[i]

        J = res_emg[1].y[:Nx, i]
        M = res_emg[1].y[Nx:2*Nx, i]
        A = res_emg[1].y[2*Nx:3*Nx, i]
        P = res_emg[1].y[3*Nx:, i]

        ax.plot(res_emg[2]["X"], J, label=figure_helper.par3add_labels[0],
                color=figure_helper.par3add_colours[0],
                linewidth=figure_helper.line_width,
               )
        ax.plot(res_emg[2]["X"], M, label=figure_helper.par3add_labels[1],
                color=figure_helper.par3add_colours[1],
                linewidth=figure_helper.line_width,
               )
        ax.plot(res_emg[2]["X"], A, label=figure_helper.par3add_labels[2],
                color=figure_helper.par3add_colours[2],
                linewidth=figure_helper.line_width,
               )
        ax.plot(res_emg[2]["X"], P, label=figure_helper.par3add_labels[3],
                color=figure_helper.par3add_colours[3],
                linewidth=figure_helper.line_width,
               )

        ax.tick_params(which="both", labelsize=figure_helper.font_size)
        ax.set_title(f"t={plot_times[i]}", fontsize=figure_helper.font_size)
        ax.text(0.05, 1.02, ["A","B","C","D"][i], transform=ax.transAxes, ha="center", fontsize=figure_helper.font_size)
        ax.text(0.9, 1.02, f"p={metric_functions.polarity_measure(res_emg[2]["X"], M+A, P, Nx):.2f}",
                transform=ax.transAxes, ha="center", fontsize=figure_helper.label_font_size)


    axs[0].legend(loc="upper left", fontsize=figure_helper.label_font_size)
    plt.xticks([0, 70])
    plt.yticks([0, 1, 2, 3, 4])

    fig.set_size_inches(15,3)
    plt.savefig("emergence_par3add_timeline.pdf", bbox_inches="tight")
    
    
    plt.show()


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()

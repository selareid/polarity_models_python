# generate illustration scribble downregulation (via rhoP), secretory phase

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import matplotlib
from matplotlib import pyplot as plt
from src import model_task_handler
from ..models import MODELS, model_to_module, metric_functions
import numpy as np
from src import figure_helper


def v_func_zero(kvals, x, t):
    return 0


Module_Par3Add = model_to_module(MODELS.PAR3ADD)


def main():
    Nx = 100

    plot_times = [0, 20000]

    initial_condition_1 = np.array([0]*(Nx//2) + [1.1]*(Nx-Nx//2)
                                   + [0]*(Nx//2) + [1.5]*(Nx-Nx//2)
                                   + [0]*(Nx//2) + [0.5]*(Nx-Nx//2)
                                   + [4.3]*(Nx//2) + [0]*(Nx-Nx//2))

    res_baseline = model_task_handler.load_or_run("maintenance_par3add", [(MODELS.PAR3ADD, {"Nx": Nx,
                                                                          "v_func": v_func_zero,
                                                                          "initial_condition": initial_condition_1,
                                                                          "label": "par3add maintenance run"})
                                                                         ])[0]

    initial_condition_2 = res_baseline[1].y[:, -1]

    tasks = [(MODELS.PAR3ADD, {"Nx": Nx,
                               "v_func": v_func_zero,
                               "tL": plot_times[-1],
                               "t_eval": plot_times,
                               "initial_condition": initial_condition_2,
                               "label": "rho_P=0.61 par3add maintenance",
                               "rho_P": 0.61,
                              }),
             (MODELS.PAR3ADD, {"Nx": Nx,
                               "v_func": v_func_zero,
                               "tL": plot_times[-1],
                               "t_eval": plot_times,
                               "initial_condition": initial_condition_2,
                               "label": "rho_P=0.60 par3add maintenance",
                               "rho_P": 0.60,
                              }),
            ]

    all_res = model_task_handler.load_or_run("maintenance_par3add", tasks)
    res = all_res[0]

    # plot
    fig, axs = plt.subplots(nrows=1, ncols=len(plot_times) + 1, sharex=True, sharey=True)

    for i in range(len(plot_times) + 1):
        # plot subfigure
        
        if i == len(plot_times):
            ax = axs[i]
            i = len(plot_times) - 1
            res = all_res[1]
        else:
            assert res[1].t[i] == plot_times[i]
            ax = axs[i]

        J = res[1].y[:Nx, i]
        M = res[1].y[Nx:2*Nx, i]
        A = res[1].y[2*Nx:3*Nx, i]
        P = res[1].y[3*Nx:, i]

        ax.plot(res[2]["X"], J, label=figure_helper.par3add_labels[0],
                color=figure_helper.par3add_colours[0],
                linewidth=figure_helper.line_width,
               )
        ax.plot(res[2]["X"], M, label=figure_helper.par3add_labels[1],
                color=figure_helper.par3add_colours[1],
                linewidth=figure_helper.line_width,
               )
        ax.plot(res[2]["X"], A, label=figure_helper.par3add_labels[2],
                color=figure_helper.par3add_colours[2],
                linewidth=figure_helper.line_width,
               )
        ax.plot(res[2]["X"], P, label=figure_helper.par3add_labels[3],
                color=figure_helper.par3add_colours[3],
                linewidth=figure_helper.line_width,
               )

        ax.tick_params(which="both", labelsize=figure_helper.font_size)
        ax.set_title(rf"$\rho_P$={[res_baseline[2]['rho_P'], res[2]['rho_P']][i]}"+r" $\text{μm}^{-3}$", fontsize=figure_helper.font_size)
        # ax.text(0.05, 1.02, ["A","B","C","D"][i], transform=ax.transAxes, ha="center", fontsize=figure_helper.font_size)
        ax.text(0.9, 1.02, f"p={metric_functions.polarity_measure(res[2]["X"], M+A, P, Nx):.2f}",
                transform=ax.transAxes, ha="center", fontsize=figure_helper.label_font_size)
        ax.set_xlabel(figure_helper.xlabel, fontsize=figure_helper.font_size)

    # overall fig plot stuff
    axs[2].legend(loc="upper right", fontsize=figure_helper.font_size)
    axs[0].set_ylabel(figure_helper.ylabel, fontsize=figure_helper.font_size)
    plt.xticks([0, 70])
    plt.yticks([0, 1, 2, 3, 4])

    # fig.set_size_inches(16,6)
    fig.set_size_inches(16,4)
    plt.savefig(f"scribble_downreg_fig_{res[0]}_rhop.pdf", bbox_inches="tight")
    
    plt.show()


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()

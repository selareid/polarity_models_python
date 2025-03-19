# generate illustrations of polarity metric using goehring

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import matplotlib
from matplotlib import pyplot as plt
from src import model_task_handler
from ..models import MODELS, metric_functions
from src import figure_helper


def v_func_zero(kvals, x, t):
    return 0


def main():
    Nx = 100
    plot_times = [0, 9000]

    polarisation_1_state = ([0]*(Nx//2) + [3]*(Nx-Nx//2)
                            + [1]*(Nx//2) + [0]*(Nx-Nx//2))

    # run to reach apically-dominant steady-state
    start_initial_condition = [1]*Nx+[0]*Nx
    task_hom = [(MODELS.GOEHRING,
                {
                    "Nx": Nx,
                    "t_eval": [0, 9000],
                    "v_func": v_func_zero,
                    "initial_condition": start_initial_condition,
                    "label": "goehring_get_a_dominant_ss"
                })]
    res_hom = model_task_handler.load_or_run("goehring_get_dominant_ss", task_hom)[0]
    initial_condition = res_hom[1].y[:,-1]


    # run to polarise
    task_emergence = [(MODELS.GOEHRING,
                      {
                          "Nx": Nx,
                          "t_eval": plot_times,
                          "initial_condition": initial_condition,
                          "label": "goehring_emergence"
                      })]
    res_emg = model_task_handler.load_or_run("goehring_emergence", task_emergence)[0]

    # plot
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True)

    for i in range(0,3):
        # select data
        if i == 0:
            A = polarisation_1_state[:Nx]
            P = polarisation_1_state[Nx:]
        elif i == 1:
            A = res_emg[1].y[:Nx, 1]
            P = res_emg[1].y[Nx:, 1]
        else:
            A = res_emg[1].y[:Nx, 0]
            P = res_emg[1].y[Nx:, 0]

        # plot subfigure
        ax = axs[i]

        ax.plot(res_emg[2]["X"], A,
                color=figure_helper.par3add_colours[0],
                linewidth=figure_helper.line_width,
                label=figure_helper.goehring_labels[0],
                )
        ax.plot(res_emg[2]["X"], P,
                color=figure_helper.par3add_colours[1],
                linewidth=figure_helper.line_width,
                label=figure_helper.goehring_labels[1],
                )

        p_m = metric_functions.polarity_measure(res_emg[2]["X"], A, P, Nx)

        ax.tick_params(which="both", labelsize=figure_helper.font_size)
        ax.set_title(f"Polarisation {p_m:.2f}", fontsize=figure_helper.font_size)
        ax.set_xlabel("x", fontsize=figure_helper.font_size)

        # if i == 0:
        #     ax.set_ylabel("A/P", fontsize=figure_helper.font_size)


    axs[0].legend(loc="upper left", fontsize=figure_helper.font_size, handlelength=4)
    
    plt.xticks([0, 70])
    plt.yticks([0, 1, 2, 3, 4])
    fig.set_size_inches(16,4)
    plt.savefig("polarisation_metric_illustration_fig.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()

# generate figures for maintenance simulation

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import matplotlib
from matplotlib import pyplot as plt
from src import model_task_handler
from ..models import MODELS, metric_functions
import numpy as np
from src import figure_helper


def v_func_zero(kvals, x, t):
    return 0


def main():
    Nx = 100
    plot_times = [0, 25, 9000]

    
    initial_conditions = {
        MODELS.GOEHRING: np.array([0]*(Nx//2) + [1.9]*(Nx-Nx//2)
                                  + [4.25]*(Nx//2) + [0]*(Nx-Nx//2)),
        MODELS.PAR3ADD: np.array([0]*(Nx//2) + [1.1]*(Nx-Nx//2)
                                 + [0]*(Nx//2) + [1.5]*(Nx-Nx//2)
                                 + [0]*(Nx//2) + [0.5]*(Nx-Nx//2)
                                 + [4.3]*(Nx//2) + [0]*(Nx-Nx//2)),
        }

    tasks = [(MODELS.GOEHRING,
              {
                  "Nx": Nx,
                  "t_eval": plot_times,
                  "v_func": v_func_zero,
                  "initial_condition": initial_conditions[MODELS.GOEHRING],
                  "label": "goehring_maintenance"
              }),
             (MODELS.PAR3ADD,
              {
                  "Nx": Nx,
                  "t_eval": plot_times,
                  "v_func": v_func_zero,
                  "initial_condition": initial_conditions[MODELS.PAR3ADD],
                  "label": "par3add_maintenance"
              }),
            ]
    all_res = model_task_handler.load_or_run("maintenance_both_models", tasks)

    for res in all_res:
        fig, axs = plt.subplots(nrows=1, ncols=len(plot_times), sharex=True, sharey=True)

        for i in range(len(plot_times)):
            ax = axs[i]
            
            match res[0]:
                case MODELS.GOEHRING:
                    ax.plot(res[2]["X"], res[1].y[:Nx, i],
                            color=figure_helper.goehring_colours[0],
                            linewidth=figure_helper.line_width,
                            label=figure_helper.goehring_labels[0],
                            )
                    ax.plot(res[2]["X"], res[1].y[Nx:, i],
                            color=figure_helper.goehring_colours[1],
                            linewidth=figure_helper.line_width,
                            label=figure_helper.goehring_labels[1],
                            )
                    p_m = metric_functions.polarity_measure(res[2]["X"],
                                                            res[1].y[:Nx, i],
                                                            res[1].y[Nx:, i],
                                                            Nx)

                case MODELS.PAR3ADD:
                    ax.plot(res[2]["X"], res[1].y[:Nx, i],
                            color=figure_helper.par3add_colours[0],
                            linewidth=figure_helper.line_width,
                            label=figure_helper.par3add_labels[0],
                            )
                    ax.plot(res[2]["X"], res[1].y[Nx:2*Nx, i],
                            color=figure_helper.par3add_colours[1],
                            linewidth=figure_helper.line_width,
                            label=figure_helper.par3add_labels[1],
                            )
                    ax.plot(res[2]["X"], res[1].y[2*Nx:3*Nx, i],
                            color=figure_helper.par3add_colours[2],
                            linewidth=figure_helper.line_width,
                            label=figure_helper.par3add_labels[2],
                            )
                    ax.plot(res[2]["X"], res[1].y[3*Nx:, i],
                            color=figure_helper.par3add_colours[3],
                            linewidth=figure_helper.line_width,
                            label=figure_helper.par3add_labels[3],
                            )

                    p_m = metric_functions.polarity_measure(res[2]["X"],
                                                            res[1].y[Nx:2*Nx, i]+res[1].y[2*Nx:3*Nx, i],
                                                            res[1].y[3*Nx:, i],
                                                            Nx)
            
            ax.tick_params(which="both", labelsize=figure_helper.font_size)
            ax.set_title(f"t={plot_times[i]}", fontsize=figure_helper.font_size)
            # ax.text(0.05, 1.02, ["A","B","C","D"][i], transform=ax.transAxes, ha="center", fontsize=figure_helper.font_size)
            ax.text(0.9, 1.02, f"p={p_m:.2f}",
                    transform=ax.transAxes, ha="center", fontsize=figure_helper.label_font_size)
            ax.set_xlabel("x", fontsize=figure_helper.font_size)


        axs[0].legend(loc="upper left", fontsize=figure_helper.label_font_size, borderaxespad=1.5)
        plt.xticks([0, 70])
        plt.yticks([0, 1, 2, 3, 4])

        fig.set_size_inches(16,4)
        plt.savefig(f"maintenance_fig_{res[0]}.pdf", bbox_inches="tight")
        
        plt.show()


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()

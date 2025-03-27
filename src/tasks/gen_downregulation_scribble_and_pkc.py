# Generate figures for reduction in
# parameters rhoA and rhoP

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


Nx = 100
base_rhoA = 1.56
base_rhoP = 1

# order is (rhoP, rhoA)
line_colours = ["blue", "green", "purple", "orange"]
lines_to_panel = [
                    [(base_rhoP, base_rhoA), (0.75, 0.78), (0.4, 0.78), (0.15, 0.78)],
                    [(base_rhoP, base_rhoA), (base_rhoP, 1.01), (base_rhoP, 0.46)],
                    [(base_rhoP, base_rhoA), (0.65, base_rhoA), (0.35, base_rhoA)],
                ]


def main():
    # get tasks by rhoA/pkc modifier value
    tasks = get_tasks()

    print(f"There are {len(tasks)} sets of tasks to run")

    results_by_rhoap = {}

    for k in tasks.keys():
        print(f"running for rhoA={k}")
        results_by_rhoap[k] = model_task_handler.load_or_run(f"gen_down_rhoA{k}", tasks[k])

    # find results to plot in panels
    panelled_res = [[] for _i in range(len(lines_to_panel))]

    for res_group_i in results_by_rhoap.keys():
        res_group = results_by_rhoap[res_group_i]
        
        for res in res_group:
            # check if point is 'similar' to a point we want to panel
            for panel_i in range(len(lines_to_panel)):
                want_to_panel = lines_to_panel[panel_i]
                for x_i in range(len(want_to_panel)):
                    x = want_to_panel[x_i]

                    dist = (x[0]-res[2]["rho_P"])**2 + (x[1]-res[2]["rho_A"])**2
                    if dist < 0.0001:  # close enough
                        # add result to our plotting list
                        # we use a tuple for later sorting purposes
                        # this will get removed after sorting
                        panelled_res[panel_i].insert(x_i, (x_i,res))
                        break

    # make sure panelled_res is sorted
    for i in range(len(panelled_res)):
        panelled_res[i].sort()
        
        # strip the sorting key from panelled_res
        panelled_res[i] = [x[1] for x in panelled_res[i]]

    # plot heatmap grid with all parameter values
    plot_all_variations(results_by_rhoap, panelled_res)

    # plot endtime for points/lines shown on grid
    for line_i in range(len(panelled_res)):
        plot_panels(panelled_res[line_i], line_colours[line_i])


def plot_panels(res_to_plot: list, colour=str):
    fig, axs = plt.subplots(nrows=1, ncols=max(2,len(res_to_plot)),
                            sharex=True, sharey=True)

    for i in range(len(res_to_plot)):
        ax = axs[i]
        res = res_to_plot[i]

        X = res[2]["X"]
        J = res[1].y[:Nx, -1]
        M = res[1].y[Nx:2*Nx, -1]
        A = res[1].y[2*Nx:3*Nx, -1]
        P = res[1].y[3*Nx:, -1]

        p_m = metric_functions.polarity_measure(X,
                                                M+A,
                                                P,
                                                Nx)
        ax.plot(X, J,
                color=figure_helper.par3add_colours[0],
                linewidth=figure_helper.line_width,
                label=figure_helper.par3add_labels[0],
                )
        ax.plot(X, M,
                color=figure_helper.par3add_colours[1],
                linewidth=figure_helper.line_width,
                label=figure_helper.par3add_labels[1],
                )
        ax.plot(X, A,
                color=figure_helper.par3add_colours[2],
                linewidth=figure_helper.line_width,
                label=figure_helper.par3add_labels[2],
                )
        ax.plot(X, P,
                color=figure_helper.par3add_colours[3],
                linewidth=figure_helper.line_width,
                label=figure_helper.par3add_labels[3],
                )

        ax.tick_params(which="both", labelsize=figure_helper.font_size)
        ax.set_title(rf"$\rho_A=${res[2]['rho_A']:.2f},"+rf"$\rho_P=${res[2]['rho_P']:.2f}",
                     fontsize=figure_helper.font_size)
        ax.text(0.05, 1.02, ["A","B","C","D", "E", "F", "G"][i],
                transform=ax.transAxes, ha="center",
                fontsize=figure_helper.font_size,
                color=colour)
        ax.text(1, 1.02, f"p={p_m:.2f}",
                transform=ax.transAxes, ha="center",
                fontsize=figure_helper.label_font_size)
        ax.set_xlabel(figure_helper.xlabel,
                      fontsize=figure_helper.font_size)

    axs[-1].legend(
                   # loc="upper left",
                  fontsize=figure_helper.label_font_size,
                  borderaxespad=1.5)
    axs[0].set_ylabel(figure_helper.ylabel,
                      fontsize=figure_helper.font_size)
    plt.xticks([0, 70])
    plt.yticks([0, 1, 2, 3, 4])

    fig.set_size_inches(16 if len(res_to_plot) < 4 else 20, 4)
    plt.savefig(f"gen_downregulation_scribble_and_pkc_{colour}.pdf", bbox_inches="tight")


def plot_all_variations(res_by_rhoA, panelled_res_list):
    plt.figure()

    # scatter plot polarisation for all result parameter values
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["grey", "yellow"])
    norm = plt.Normalize(0, 1)

    for p_a in res_by_rhoA:
        for res in res_by_rhoA[p_a]:
            if res[1] != "FAILURE":
                rho_A = res[2]["rho_A"]
                rho_P = res[2]["rho_P"]

                polarisation = metric_functions.polarity_measure(res[2]["X"],
                                                                 res[1].y[Nx:2*Nx, -1] + res[1].y[2*Nx:3*Nx, -1],
                                                                 res[1].y[3*Nx:, -1],
                                                                 Nx)

                plt.scatter(rho_P, rho_A,
                            c=polarisation,
                            cmap=cmap,
                            norm=norm,
                            marker='o',
                            s=100)

    cbar = plt.colorbar()
    cbar.set_label("polarisation")

    # plot selected lines of variations
    for i in range(len(panelled_res_list)):
        panelled_res = panelled_res_list[i]
        panelled_res_plot = ([], [])
        for res in panelled_res:
            panelled_res_plot[0].append(res[2]["rho_A"])
            panelled_res_plot[1].append(res[2]["rho_P"])
        plt.plot(panelled_res_plot[1], panelled_res_plot[0], "--o", color=line_colours[i])

    # mark baseline point
    plt.scatter(base_rhoP, base_rhoA, c="black", marker=".", s=150, zorder=2)

    plt.xlabel(r"$\rho_P$")
    plt.ylabel(r"$\rho_A$")

    plt.savefig("gen_downregulation_scribble_and_pkc_grid.pdf", bbox_inches="tight")


def get_tasks():
    # list of parameter changes we consider
    percent_changes = np.arange(0, 1.25 + 0.05, 0.05).tolist()

    plot_times = [0, 300, 9000, 19000, 20000]  # more points than needed helps debugging

    initial_condition = np.array([0]*(Nx//2) + [1.1]*(Nx-Nx//2)
                                 + [0]*(Nx//2) + [1.5]*(Nx-Nx//2)
                                 + [0]*(Nx//2) + [0.5]*(Nx-Nx//2)
                                 + [4.3]*(Nx//2) + [0]*(Nx-Nx//2))

    tasks_by_rhoA = {}  # tasks indexed by rhoA percent change

    for p_a in percent_changes:
        p_a_array = []
        for p_p in percent_changes:
            # add task for current parameter pair
            p_a_array.append((MODELS.PAR3ADD, {"Nx": Nx,
                                               "t_eval": plot_times,
                                               "tL": plot_times[-1],
                                               "v_func": v_func_zero,
                                               "initial_condition": initial_condition,
                                               "rho_A": base_rhoA*p_a,
                                               "rho_P": base_rhoP*p_p,
                                               "label": f"rhoA:{p_a};rhoP{p_p}",
                                               }))
        tasks_by_rhoA[p_a] = p_a_array

    return tasks_by_rhoA


if __name__ == '__main__':
    matplotlib.use("Agg")
    main()
    plt.show()

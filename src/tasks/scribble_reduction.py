# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from src.models import model_to_module, MODELS
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from .. import model_task_handler
from src.tasks.variation_task_helper import generate_tasks, split_baseline_from_results, run_grouped_tasks, \
    get_variation_multiplier, get_xticks

model_module = model_to_module(MODELS.GOEHRING)

# COLOR_APAR = (0, 200/255, 200/255)
# COLOR_PPAR = (250/255, 0, 0)
COLOR_APAR = (0, 150/255, 150/255)
COLOR_PPAR = (230/255, 0, 0)
BASELINE_COLOR = [(0, 100/255, 100/255), (180/255, 0, 0)]
# EXTRA_COLOR = [(0, 40/255, 175/255), (200/255, 0, 100/255)]

LINE_WIDTH=5


def v_func_def(kvals, x, t):
    v_time = 600
    time_factor = 1 / np.maximum(1, t / 10 - v_time / 10)

    center = kvals["xL"] / 4
    sd = np.minimum(center / 4, (kvals["xL"] - center) / 4)
    peak = 0.1

    return time_factor * peak * np.exp(-(x - center) ** 2 / (2 * sd ** 2))


def v_func_zero(kvals, x, t):
    return 0


def get_default_parameters(Nx, tL, v_func=v_func_zero):
    return {"label": "goehring",
            "key_varied": "",
            "points_per_second": 0.01,

            # General Setup Variables
            "Nx": Nx,  # number of length steps
            "L": 134.6,  # length of region
            "x0": 0,
            "xL": 67.3,  # L / 2
            "t0": 0,
            "tL": tL,

            # Model parameters and functions
            # "psi": 0.174,

            # "D_A": 0.28,
            # "D_P": 0.15,

            # "k_onA": 8.58 * 10 ** (-3),
            "k_onP": 4.74 * 10 ** (-2),

            # "k_offA": 5.4 * 10 ** (-3),
            "k_offP": 7.3 * 10 ** (-3),

            # "k_AP": 0.190,
            "k_PA": 2.0,

            # "rho_A": 1.56,
            # "rho_P": 1.0,

            # "alpha": 1,
            # "beta": 2,

            # R_X
            # Xbar
            # "A_cyto": default_A_cyto,
            # "P_cyto": default_P_cyto,
            "v_func": v_func
            }


def test_polarity_maintenance(Nx=50, tL=3000, initial_condition=None, time_addition=0, extra_plot=None):
    if initial_condition is None:
        initial_condition = [0] * (Nx // 2) + [1] * (Nx - Nx // 2) + [1] * (Nx // 2) + [0] * (Nx - Nx // 2) # polarised

    variation_multipliers, index_for_100x = get_variation_multiplier("scribble")
    xticks = get_xticks(variation_multipliers, index_for_100x)

    tasks = generate_tasks(MODELS.GOEHRING, variation_multipliers,
                           get_default_parameters(Nx=Nx, tL=tL, v_func=v_func_zero),
             {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity maintenance")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    baseline, results_by_variable = split_baseline_from_results(MODELS.GOEHRING, run_grouped_tasks(tasks), index_for_100x, extra_plot=extra_plot)

    print("Plotting")
    # model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL), x_axis_labels=xticks)
    # model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL), x_axis_labels=xticks, xlim=["60%", "140%"])
    
    filtered_sols_by_variable = []
    filtered_kvals_by_variable = []

    for sol_list, kvals_list in results_by_variable:
        sols_filtered = []
        kvals_filtered = []

        for i in range(0, len(sol_list), 1) if kvals_list[0]["key_varied"] in ["k_offP", "k_PA"] else range(len(sol_list)-1, -1, -1):
            sol, kvals = sol_list[i], kvals_list[i]
            if kvals["key_varied"] == "k_offP" and kvals["variation_multiplier"] >= 0.999 \
                    or kvals["key_varied"] == "k_onP" and kvals["variation_multiplier"] <= 1.001 \
                    or kvals["key_varied"] == "k_PA" and kvals["variation_multiplier"] >= 0.999:
                sols_filtered.append(sol)
                kvals_filtered.append(kvals)

        filtered_sols_by_variable.append(sols_filtered)
        filtered_kvals_by_variable.append(kvals_filtered)

        # multi_plot_for_poster(sols_filtered, kvals_filtered)
    
    # multi_plot_all_variables(filtered_sols_by_variable, filtered_kvals_by_variable, time_addition=time_addition)
    #two_plot_for_poster(filtered_sols_by_variable, filtered_kvals_by_variable, time_addition=time_addition)


def multi_plot_for_poster(sol_list, kvals_list, plot_A=True, plot_P=True, rescale=True):
    X, Nx, x0, xL, tL, key_varied = kvals_list[0]["X"], kvals_list[0]["Nx"], kvals_list[0]["x0"], kvals_list[0]["xL"], kvals_list[0]["tL"], kvals_list[0]["key_varied"]

    plt.figure()
    ax = plt.subplot()

    max_aPar_unscaled = np.max([sol.y[:Nx] for sol in sol_list])
    max_pPar_unscaled = np.max([sol.y[Nx:] for sol in sol_list])
    rescaleFactor = 1 if not rescale else 1/max_pPar_unscaled

    for i in np.arange(0, len(sol_list)):
        sol = sol_list[i]
        kvals = kvals_list[i]

        annote_shift = 0.5 * (i % 3 - 1)

        if plot_A:
            ax.plot(X, sol.y[:Nx, -1]*rescaleFactor, color=COLOR_APAR, alpha=1-i/(1.1*len(sol_list)))
            ax.annotate(format(f"{kvals['variation_multiplier'] * 100:.1f}").rstrip('0').rstrip('.')+"%",
                        (X[Nx//2], sol.y[Nx//2, -1]*rescaleFactor),
                            xytext=(annote_shift+X[Nx//2], sol.y[Nx//2, -1]*rescaleFactor), color=COLOR_APAR, alpha=1)
        if plot_P:
            ax.plot(X, sol.y[Nx:, -1]*rescaleFactor, color=COLOR_PPAR, alpha=1-i/(1.1*len(sol_list)))
            ax.annotate(format(f"{kvals['variation_multiplier'] * 100:.1f}").rstrip('0').rstrip('.')+"%",
                        (X[Nx//5], sol.y[Nx+Nx//5, -1]*rescaleFactor),
                            xytext=(annote_shift+X[Nx//5], sol.y[Nx+Nx//5, -1]*rescaleFactor), color=COLOR_PPAR, alpha=1)


    # TODO TODO
    # ticks
    ax.set_xticks([x0, xL/2, xL])
    ax.set_xticklabels(["0", "mid", "L"])
    ax.set_yticks([0, max_aPar_unscaled*rescaleFactor, max_pPar_unscaled*rescaleFactor])
    ax.set_yticklabels(["0", "max aPar", "max pPar"])

    # ax.text(0.1, 1.05, f"t={sol_list[0].t[-1]}", transform=ax.transAxes, ha="center")  # timestamp
    # ax.text(1, 1.05, label, transform=ax.transAxes, ha="center")  # label

    # ax.set(xlim=[kvals["x0"], kvals["xL"]],
    #        ylim=[np.min([sol.y[:, -1] for sol in sol_list]) - 0.05, np.max([sol.y[:, -1] for sol in sol_list]) + 0.05],
    #        xlabel="x", ylabel="A/P")
    ax.title.set_text(format(f"N={Nx},t={tL},key_varied={key_varied},scaled={rescale}"))

    # A, P legend
    handles = [Line2D([], [], c=COLOR_APAR), Line2D([], [], c=COLOR_PPAR)]
    ax.legend(handles, ["Anterior Proteins", "Posterior Proteins"])
    # TODO TODO

    plt.show(block=False)


def multi_plot_all_variables(sols_by_variable, kvals_by_variable, plot_A=True, plot_P=True, time_addition=0):
    plt.figure()
    ax = plt.subplot()

    left_ticks = []
    right_ticks = []

    base_plotted = False
    baseline_sol = None

    for i in range(len(sols_by_variable)):
        sol_list, kvals_list = sols_by_variable[i], kvals_by_variable[i]
        X, Nx, x0, xL, tL, key_varied = kvals_list[0]["X"], kvals_list[0]["Nx"], kvals_list[0]["x0"], kvals_list[0]["xL"], kvals_list[0]["tL"], kvals_list[0]["key_varied"]

        for j in range(len(sol_list)):
            sol, kvals = sol_list[j], kvals_list[j]

            is_base = kvals["variation_multiplier"] == 1

            if is_base and base_plotted: continue

            if is_base: marker = "solid"
            else: marker = ["--",":","-."][i]

            line_alpha = 1-j/(1.3*len(sol_list))

            annote = f"{kvals['variation_multiplier'] * 100:.0f}"+"%" + kvals["key_varied"] if not is_base else "baseline"
            annote = annote.replace("k_PA", "$k_{PA}$")
            annote = annote.replace("k_onP", "$k_{\\text{on},P}$")


            if plot_A:
                ax.plot(X, sol.y[:Nx, -1], color=COLOR_APAR if not is_base else BASELINE_COLOR[0], linestyle=marker, alpha=line_alpha, linewidth=LINE_WIDTH, zorder=0)
                # ax.annotate(,
                #         (X[Nx//2], sol.y[Nx//2, -1]),
                #             xytext=(X[Nx//2], sol.y[Nx//2, -1]),
                #             color=COLOR_APAR, alpha=1)
                right_ticks.append((sol.y[Nx-1, -1],
                                    ("\n" if not is_base else "") + ("\n" if not is_base and annote not in "80%$k_{\\text{on},P}$" else "") + (annote if annote not in ["135%$k_{PA}$", "65%$k_{\\text{on},P}$"] else "\n"+annote),
                                    i if not is_base else -1)
                                   )
            if plot_P:
                ax.plot(X, sol.y[Nx:, -1], color=COLOR_PPAR if not is_base else BASELINE_COLOR[1], linestyle=marker, alpha=line_alpha, linewidth=LINE_WIDTH, zorder=1)
                # ax.annotate(annote,
                #         (X[0], sol.y[Nx+0, -1]),
                #             xytext=(X[0]-2, sol.y[Nx+0, -1] if sol.y[Nx+0, -1] > 0.01 else sol.y[Nx+0, -1] + ((j+i)%3-1)/10 ),
                #             color=(i%2, ((i)%3)/2, 0) if not is_base else COLOR_PPAR, alpha=1)
                left_ticks.append((sol.y[Nx+0, -1],
                                   annote if annote not in ["65%$k_{\\text{on},P}$", "170%$k_{PA}$"] else "\n"+annote if annote != "65%$k_{\\text{on},P}$" else "\n\n\n"+annote,
                                   i if not is_base else -1)
                                  )


            if is_base:
                baseline_sol = sol.y
                base_plotted = True
    

    ax.text(0.1, 1.025, f"t={tL + time_addition:.0f}", transform=ax.transAxes, ha="center") # time value

    left_ticks.sort()
    right_ticks.sort()

    # ticks
    ax.set_xticks([x0, xL/2, xL])
    ax.set_xticklabels(["0", "mid", "L"])

    # ax.set_yticks([0, baseline_sol[Nx, -1]])  # TODO TODO
    # ax.set_yticklabels(["0", "baseline pPar"])
    ax.set_yticks([0] + [x[0] for x in left_ticks], ["0"] + [x[1] for x in left_ticks])
    # ax.set_yticklabels()

    y2 = ax.twinx()
    y2.set_ylim(ax.get_ylim())
    y2.set_yticks([0] + [x[0] for x in right_ticks], ["0"] + [x[1] for x in right_ticks])
    # y2.set_yticklabels(["0", "baseline aPar"])

    # for i in range(len(left_ticks)):
    #     _, s_l, j_l = left_ticks[i]
    #     ax.get_yticklabels()[i+1].set_color([COLOR_PPAR,COLOR_PPAR,(0,0,0)][j_l])
    #
    #     _, s_r, j_r = right_ticks[i]
    #     y2.get_yticklabels()[i+1].set_color([COLOR_APAR,COLOR_APAR,(0,0,0)][j_r])

    # A, P legend
    handles = [Line2D([], [], c=COLOR_APAR, linewidth=LINE_WIDTH), Line2D([], [], c=COLOR_PPAR, linewidth=LINE_WIDTH)]
    ax.legend(handles, ["Anterior", "Posterior"], prop={'size': 18})

    plt.show(block=False)


def two_plot_for_poster(sols_by_variable, kvals_by_variable, time_addition=0):
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
    # upperYTick = np.max(initial_sol.y)
    # # Single Figure multi-plot
    # model_module.plot_timestep(initial_sol, initial_kvals, 0, upperYTick=upperYTick, show_legend=False, ax=axs[0])
    # model_module.plot_timestep(initial_sol, initial_kvals, 1, upperYTick=upperYTick, show_legend=False, left_ticks=False, ax=axs[1])
    # COLOR_APAR = (0, 150 / 255, 150 / 255)
    # COLOR_PPAR = (230 / 255, 0, 0)
    # LINE_WIDTH = 5

    assert len(sols_by_variable) == 2

    for i in range(len(sols_by_variable)):
        sol_list, kvals_list = sols_by_variable[i], kvals_by_variable[i]
        X, Nx, x0, xL, tL, key_varied = kvals_list[0]["X"], kvals_list[0]["Nx"], kvals_list[0]["x0"], kvals_list[0]["xL"], kvals_list[0]["tL"], kvals_list[0]["key_varied"]
        ax = axs[i]

        for j in range(len(sol_list)):
            sol, kvals = sol_list[j], kvals_list[j]

            is_base = kvals["variation_multiplier"] == 1

            ax.plot(X, sol.y[:Nx, -1], color=BASELINE_COLOR[0] if is_base else COLOR_APAR,
                    linewidth=LINE_WIDTH, zorder=0+(1 if is_base else 0))  # A
            ax.plot(X, sol.y[Nx:, -1], color=BASELINE_COLOR[1] if is_base else COLOR_PPAR,
                    linewidth=LINE_WIDTH, zorder=10+(1 if is_base else 0))  # P

        ax.set_title(key_varied)


    fig.text(0.25, 1.025, f"t={sol.t[-1] + time_addition:.0f}", transform=axs[0].transAxes, ha="center") # time value
    axs[0].set_xticks([0,xL//2,xL],['0','','L'])
    axs[0].set_yticks([0,np.max(sol.y)], ['',''])

    # A, P legend
    handles = [Line2D([], [], c=COLOR_APAR, linewidth=LINE_WIDTH), Line2D([], [], c=COLOR_PPAR, linewidth=LINE_WIDTH)]
    ax.legend(handles, ["Anterior", "Posterior"])


# reducing k_on,P; increasing k_off,P
def run_scribble_variation(Nx, tL, initial_condition, time_addition=0):
    test_polarity_maintenance(Nx, tL, initial_condition, time_addition)


if __name__ == '__main__':
    Nx = 100  #100
    tL_establishment = 3000
    tL_maintenance = 12000

    # run model initially to get initial condition for the reductions
    _, initial_sol, initial_kvals = model_task_handler.run_tasks([(MODELS.GOEHRING,
                                        {**get_default_parameters(Nx, tL_establishment, v_func_def),
                                        "initial_condition": [2.828]*Nx + [0.025]*Nx,
                                        "t_eval": [0,600, 700, 800,1000,2000,3000]
                                        })])[0]

    # Plotting

    # upperYTick = np.max(initial_sol.y)

    # # Single Figure multi-plot
    # fig, axs = plt.subplots(1, 5, sharey=True, sharex=True)
    # model_module.plot_timestep(initial_sol, initial_kvals, 0, upperYTick=upperYTick, show_legend=False, ax=axs[0])
    # model_module.plot_timestep(initial_sol, initial_kvals, 1, upperYTick=upperYTick, show_legend=False, left_ticks=False, ax=axs[1])
    # model_module.plot_timestep(initial_sol, initial_kvals, 2, upperYTick=upperYTick, show_legend=False, left_ticks=False, ax=axs[2])
    # model_module.plot_timestep(initial_sol, initial_kvals, 4, upperYTick=upperYTick, show_legend=False, left_ticks=False, ax=axs[3])
    # model_module.plot_timestep(initial_sol, initial_kvals, len(initial_sol.y[0,:])-1, upperYTick=upperYTick, show_legend=True, left_ticks=False, ax=axs[4])

    # # Individual Plots
    # model_module.plot_timestep(initial_sol, initial_kvals, 0, upperYTick=upperYTick, show_legend=False)
    # model_module.plot_timestep(initial_sol, initial_kvals, 1, upperYTick=upperYTick, show_legend=False, left_ticks=False)
    # model_module.plot_timestep(initial_sol, initial_kvals, 2, upperYTick=upperYTick, show_legend=False, left_ticks=False)
    # model_module.plot_timestep(initial_sol, initial_kvals, 4, upperYTick=upperYTick, show_legend=False, left_ticks=False)
    # model_module.plot_timestep(initial_sol, initial_kvals, len(initial_sol.y[0,:])-1, upperYTick=upperYTick, show_legend=True, left_ticks=False)

    # model_module.animate_plot(initial_sol, initial_kvals)

    # Parameter Variation using previous run as IC
    initial_condition  = initial_sol.y[:, -1]
    run_scribble_variation(Nx, tL_maintenance, initial_condition, time_addition=tL_establishment)

    plt.show()
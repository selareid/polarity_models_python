# https://stackoverflow.com/questions/17053671/how-do-you-stop-numpy-from-multithreading
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from src.tasks.variation_task_helper import generate_tasks, split_baseline_from_results, run_grouped_tasks, \
    get_variation_multiplier, get_xticks
from ..models import MODELS, model_to_module

model_module = model_to_module(MODELS.GOEHRING)


def default_v_func(kvals, x, t):
    v_time = 600
    time_factor = 1 / np.maximum(1, t / 10 - v_time / 10)

    center = kvals["xL"] / 4
    sd = np.minimum(center / 4, (kvals["xL"] - center) / 4)
    peak = 0.1

    return time_factor * peak * np.exp(-(x - center) ** 2 / (2 * sd ** 2))


def v_func_zero(kvals, x, t):
    return 0


def get_default_parameters(Nx=50, tL=3000, v_func=default_v_func, filter=None):
    if filter == "A":  # Just A Stuff
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                "D_A": 0.28,

                "k_onA": 8.58 * 10 ** (-3),

                "k_offA": 5.4 * 10 ** (-3),

                "k_AP": 0.190,

                "rho_A": 1.56,

                "v_func": v_func
         }
    elif filter == "P":  # Just P Stuff
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                "D_P": 0.15,

                "k_onP": 4.74 * 10 ** (-2),

                "k_offP": 7.3 * 10 ** (-3),

                "k_PA": 2.0,

                "rho_P": 1.0,

                "v_func": v_func
                }
    elif filter == "generic":  # not A, not P stuff
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                "psi": 0.174,

                "alpha": 1,
                "beta": 2,

                "v_func": v_func
                }
    elif filter == "S":  # the ones relevant to scribble reduction
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                # "D_P": 0.15,

                "k_onP": 4.74 * 10 ** (-2),

                "k_offP": 7.3 * 10 ** (-3),

                "k_PA": 2.0,

                # "rho_P": 1.0,

                "v_func": v_func
                }
    elif filter == "S_TEST":  # the ones relevant to scribble reduction
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                # "D_P": 0.15,

                # "k_onP": 4.74 * 10 ** (-2),

                "k_offP": 7.3 * 10 ** (-3),

                # "k_PA": 2.0,

                # "rho_P": 1.0,

                "v_func": v_func
                }
    else:  # All
        return {"key_varied": "",
                "label": "goehring",
                "points_per_second": 0.01,

                # General Setup Variables
                "Nx": Nx,  # number of length steps
                "L": 134.6,  # length of region
                "x0": 0,
                "xL": 67.3,  # L / 2
                "t0": 0,
                "tL": tL,

                # Model parameters and functions
                "psi": 0.174,

                "D_A": 0.28,
                "D_P": 0.15,

                "k_onA": 8.58 * 10 ** (-3),
                "k_onP": 4.74 * 10 ** (-2),

                "k_offA": 5.4 * 10 ** (-3),
                "k_offP": 7.3 * 10 ** (-3),

                "k_AP": 0.190,
                "k_PA": 2.0,

                "rho_A": 1.56,
                "rho_P": 1.0,

                "alpha": 1,
                "beta": 2,

                # R_X
                # Xbar
                # "A_cyto": default_A_cyto,
                # "P_cyto": default_P_cyto,
                "v_func": v_func
                }


def test_polarity_establishment(Nx=50,tL=3000, filter=None, extra_plot=None):
    # initial_condition = [0]*(Nx//3) + [2.828] * (Nx - Nx//3) + [0.025] * Nx
    initial_condition = [2.828] * Nx + [0.025] * Nx  # the values we get when running baseline with v=0

    variation_multipliers, index_for_100x = get_variation_multiplier()
    xticks = get_xticks(variation_multipliers, index_for_100x)

    tasks = generate_tasks(MODELS.GOEHRING, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL, filter=filter), {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity establishment")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    baseline, results_by_variable = split_baseline_from_results(MODELS.GOEHRING, run_grouped_tasks(tasks), index_for_100x, extra_plot=extra_plot)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks)
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"", x_axis_labels=xticks, xlim=["60%", "140%"])


def test_polarity_maintenance(Nx=50, tL=3000, filter=None, extra_plot=None):
    initial_condition = [0] * (Nx // 2) + [1] * (Nx - Nx // 2) + [1] * (Nx // 2) + [0] * (Nx - Nx // 2) # polarised

    variation_multipliers, index_for_100x = get_variation_multiplier("scribble" if filter == "S" or filter == "S_TEST" else None)
    xticks = get_xticks(variation_multipliers, index_for_100x)

    tasks = generate_tasks(MODELS.GOEHRING, variation_multipliers, get_default_parameters(Nx=Nx, tL=tL, filter=filter, v_func=v_func_zero), {"label": ".", "initial_condition": initial_condition})

    print(f"Testing polarity maintenance")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    baseline, results_by_variable = split_baseline_from_results(MODELS.GOEHRING, run_grouped_tasks(tasks), index_for_100x, extra_plot=extra_plot)

    print("Plotting")
    model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL), x_axis_labels=xticks)
    # model_module.plot_variation_sets(results_by_variable, label="Goehring Polarity Maintenance Nx="+str(Nx)+",tL="+str(tL), x_axis_labels=xticks, xlim=["60%", "140%"])

    if filter == "S" or filter == "S_TEST":  # scribble stuff
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

            multi_plot_for_poster(sols_filtered, kvals_filtered)


def multi_plot_for_poster(sol_list, kvals_list, plot_A=True, plot_P=True, rescale=True):
    X, Nx, x0, xL, tL, key_varied = kvals_list[0]["X"], kvals_list[0]["Nx"], kvals_list[0]["x0"], kvals_list[0]["xL"], kvals_list[0]["tL"], kvals_list[0]["key_varied"]

    plt.figure()
    ax = plt.subplot()

    colorAPar = (0, 230/255, 230/255)
    colorPPar = (230/255, 0, 0)

    max_aPar_unscaled = np.max([sol.y[:Nx] for sol in sol_list])
    max_pPar_unscaled = np.max([sol.y[Nx:] for sol in sol_list])
    rescaleFactor = 1 if not rescale else 1/max_pPar_unscaled

    for i in np.arange(0, len(sol_list)):
        sol = sol_list[i]
        kvals = kvals_list[i]

        annote_shift = 0.5 * (i % 3 - 1)

        if plot_A:
            ax.plot(X, sol.y[:Nx, -1]*rescaleFactor, color=colorAPar, alpha=1-i/(1.1*len(sol_list)))
            ax.annotate(format(f"{kvals['variation_multiplier'] * 100:.1f}").rstrip('0').rstrip('.')+"%",
                        (X[Nx//2], sol.y[Nx//2, -1]*rescaleFactor),
                            xytext=(annote_shift+X[Nx//2], sol.y[Nx//2, -1]*rescaleFactor), color=colorAPar, alpha=1)
        if plot_P:
            ax.plot(X, sol.y[Nx:, -1]*rescaleFactor, color=colorPPar, alpha=1-i/(1.1*len(sol_list)))
            ax.annotate(format(f"{kvals['variation_multiplier'] * 100:.1f}").rstrip('0').rstrip('.')+"%",
                        (X[Nx//5], sol.y[Nx+Nx//5, -1]*rescaleFactor),
                            xytext=(annote_shift+X[Nx//5], sol.y[Nx+Nx//5, -1]*rescaleFactor), color=colorPPar, alpha=1)


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
    handles = [Line2D([], [], c=colorAPar), Line2D([], [], c=colorPPar)]
    ax.legend(handles, ["Anterior Proteins", "Posterior Proteins"])
    # TODO TODO

    plt.show(block=False)


if __name__ == '__main__':
    # d = get_default_parameters(100, 3000, filter="S")
    extra_plot_list = [
    #         format(f"k_onP={d['k_onP'] * 0.00:.4f}"),
    #         format(f"k_onP={d['k_onP'] * 0.10:.4f}"),
    #         format(f"k_onP={d['k_onP'] * 0.50:.4f}"),
    #         format(f"k_onP={d['k_onP'] * 0.60:.4f}"),
    #         format(f"k_onP={d['k_onP'] * 0.70:.4f}"),
    #         # format(f"k_onP={d['k_onP'] * 1.3:.4f}"),
    #         # format(f"k_onP={d['k_onP'] * 1.75:.4f}"),
    #
    #         # format(f"k_offP={d['k_offP'] * 0.7:.4f}"),
    #         # format(f"k_offP={d['k_offP'] * 0.8:.4f}"),
    #         # format(f"k_offP={d['k_offP'] * 1.3:.4f}"),
    #         # format(f"k_offP={d['k_offP'] * 1.75:.4f}"),
        ]

    # test_polarity_establishment(Nx=100, filter="A", extra_plot=extra_plot_list)
    # test_polarity_establishment(Nx=100, filter="P", extra_plot=extra_plot_list)
    # test_polarity_establishment(Nx=100, filter="generic", extra_plot=extra_plot_list)
    # test_polarity_establishment(Nx=100, filter=None, extra_plot=extra_plot_list)


    # test_polarity_maintenance(Nx=100, filter="A", extra_plot=extra_plot_list)
    # test_polarity_maintenance(Nx=100, filter="P", extra_plot=extra_plot_list)

    # # Scribble Reduction
    # test_polarity_maintenance(Nx=100, tL=3000, filter="S", extra_plot=extra_plot_list)
    # test_polarity_maintenance(Nx=100, tL=6000, filter="S", extra_plot=extra_plot_list)
    # test_polarity_maintenance(Nx=10, tL=12000, filter="S", extra_plot=extra_plot_list)
    # test_polarity_maintenance(Nx=100, tL=24000, filter="S", extra_plot=extra_plot_list)

    # test_polarity_maintenance(Nx=100, filter="generic", extra_plot=extra_plot_list)
    # test_polarity_maintenance(Nx=100, filter=None, extra_plot=extra_plot_list)

    print("Finished!")
    plt.show()


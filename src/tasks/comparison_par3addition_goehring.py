# Run par3addition model and generate plots for comparison to goehring
# Note that doing parameter variations for
# sensitivity will take a long time (hours)

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
import matplotlib
from matplotlib import pyplot as plt, animation
from src import model_task_handler
from src.tasks import variation_task_helper
from ..models import MODELS, model_to_module, metric_functions
import numpy as np
from src import figure_helper


def v_func_zero(kvals, x, t):
    return 0


Module_Par3Add = model_to_module(MODELS.PAR3ADD)
Module_Goehring = model_to_module(MODELS.GOEHRING)


# base parameters
params_goehring = {
    "psi": 0.174,
    "D_A": 0.28, "D_P": 0.15,
    "k_onA": 8.58 * 10 ** (-3), "k_onP": 4.74 * 10 ** (-2),
    "k_offA": 5.4 * 10 ** (-3), "k_offP": 7.3 * 10 ** (-3),
    "k_AP": 0.190, "k_PA": 2.0,
    "rho_A": 1.56, "rho_P": 1.0,
    "alpha": 1, "beta": 2,
}

params_par3add = {
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

    "sigmaJ": 1, "sigmaM": 1, "sigmaP": 1,

    # not used in writeup
    "konA": 0,  
    "alpha": 1, "beta": 2,
   }


# parameters to vary for sensitivity test
variation_params_par3add = [
    "psi", "D_J", "D_M", "D_A", "D_P", "sigmaJ", "sigmaM", "sigmaP",
    "k1", "k2", "kAP", "kJP", "kMP", "kPA", "alpha", "beta",
    "rho_A", "rho_J", "rho_P", "konA", "konJ", "konP",
    "koffA", "koffJ", "koffP", "koffM"
]
variation_params_goehring = [
    "psi", "D_A", "D_P", "k_onA", "k_onP", "k_offA", "k_offP",
    "k_AP", "k_PA", "rho_A", "rho_P", "alpha", "beta"
]


# model-agnostic function
def get_model_variation_tasks(model: MODELS, Nx: int, tL: int, initial_condition, parameters: dict, varied_params: dict):
    variation_multipliers, index_for_100x = variation_task_helper.get_variation_multiplier()
    xticks = variation_task_helper.get_xticks(variation_multipliers, index_for_100x)

    tasks: list[list[tuple]] = variation_task_helper.generate_tasks(model, variation_multipliers, varied_params,
                                                                    {**parameters,
                                                                     "Nx": Nx, "tL": tL,
                                                                     "initial_condition": initial_condition,
                                                                     "t_eval": [0, 8500, 9000, 19000, 20000] if tL == 20000 else None,
                                                                    })

    return tasks, index_for_100x, xticks


def main():
    Nx = 100
    tL_est = 20000
    points_per_second = 0.01  # for animations

    homogeneous_ic_par3add = [1] * Nx + [1] * Nx + [1] * Nx + [0] * Nx
    homogeneous_ic_goehring = [2.828] * Nx + [0] * Nx
    parameters_par3add = {**params_par3add,
                          "points_per_second": points_per_second,
                          "Nx": Nx,
                          }
    parameters_goehring = {**params_goehring,
                           "points_per_second": points_per_second,
                           "Nx": Nx,
                           }

    # run without advection to get initial condition (IC)
    init_conds = get_homogeneous_initial_conditions(homogeneous_ic_goehring, homogeneous_ic_par3add,
                                                    parameters_goehring, parameters_par3add)

    # Run with advection using found IC
    do_establishment_run(init_conds, parameters_goehring, parameters_par3add, tL_est)

    print("Starting variations")
    # Do emergence/establishment variation test
    do_establishment_param_variation(Nx, init_conds, parameters_goehring,
                                     parameters_par3add, points_per_second, tL_est)
    # Do maintenance variation test
    do_maintenance_param_variation(Nx, {**parameters_goehring, "v_func": v_func_zero},
                                   {**parameters_par3add, "v_func": v_func_zero},
                                   points_per_second, tL_est)


# do a full run starting from the pPar dominant state
def p_dominant_run(Nx, tL, points_per_second):
    homogeneous_ic_par3add = [0]*(3*Nx) + [1]*Nx
    homogeneous_ic_goehring = [0]*Nx + [1]*Nx

    parameters_par3add = {**params_par3add,
                          "points_per_second": points_per_second,
                          "Nx": Nx,
                          }
    parameters_goehring = {**params_goehring,
                           "points_per_second": points_per_second,
                           "Nx": Nx,
                           }
    init_conds = get_homogeneous_initial_conditions(homogeneous_ic_goehring, homogeneous_ic_par3add, parameters_goehring, parameters_par3add)

    # do_establishment_run(init_conds, parameters_goehring, parameters_par3add, tL)

    # Do establishment variations
    # mostly want to see how changes in sigma change things
    do_establishment_param_variation(Nx, init_conds, parameters_goehring, parameters_par3add, points_per_second, tL)


def get_homogeneous_initial_conditions(homogeneous_ic_goehring, homogeneous_ic_par3add, parameters_goehring,
                                       parameters_par3add):
    tasks_initial_condition = [(MODELS.PAR3ADD, {
        **parameters_par3add, "initial_condition": homogeneous_ic_par3add, "label": "par3add finding IC", "tL": 5000,
        "v_func": v_func_zero}),
                               (MODELS.GOEHRING, {
                                   **parameters_goehring, "initial_condition": homogeneous_ic_goehring,
                                   "label": "goehring finding IC", "tL": 5000, "v_func": v_func_zero})
                               ]
    multi_res_initial_condition = model_task_handler.load_or_run("comparison_get_hom_ic", tasks_initial_condition)
    init_conds = {}
    for res in multi_res_initial_condition:
        match res[0]:
            case MODELS.GOEHRING:
                init_conds[MODELS.GOEHRING] = res[1].y[:, -1]
            case MODELS.PAR3ADD:
                init_conds[MODELS.PAR3ADD] = res[1].y[:, -1]

        model_to_module(res[0]).animate_plot(res[1], res[2], save_file=True, file_code="get_homogeneous_initial_condition_"+str(res[0]))

    print("Some stats of homogeneous F.P.:")
    print(f"Par3add Model: par3: {init_conds[MODELS.PAR3ADD][0]}, par3-PKC: {init_conds[MODELS.PAR3ADD][parameters_par3add["Nx"]]}, \
        cdc42-PKC: {init_conds[MODELS.PAR3ADD][2*parameters_par3add["Nx"]]}, pPar: {init_conds[MODELS.PAR3ADD][len(init_conds[MODELS.PAR3ADD]) - 1]}")
    print(f"Goehring Model: aPar: {init_conds[MODELS.GOEHRING][0]}, pPar: {init_conds[MODELS.GOEHRING][len(init_conds[MODELS.GOEHRING]) - 1]}")
    
    return init_conds


def do_establishment_run(init_conds, parameters_goehring, parameters_par3add, tL_est):
    tasks_establishment = [(MODELS.PAR3ADD, {
        **parameters_par3add, "label": "par3add establishment", "tL": tL_est,
        "initial_condition": init_conds[MODELS.PAR3ADD]}),
                           (MODELS.GOEHRING, {
                               **parameters_goehring, "label": "goehring establishment", "tL": tL_est,
                               "initial_condition": init_conds[MODELS.GOEHRING]})
                           ]

    multi_res_establishment = model_task_handler.load_or_run("comp_establishment_run", tasks_establishment)

    for res in multi_res_establishment:
        match res[0]:
            case MODELS.GOEHRING:
                g_res = res
            case MODELS.PAR3ADD:
                Module_Par3Add.animate_plot_apar_combo(res[1], res[2], save_file=True, no_par3=False)
                p_res = res

        model_to_module(res[0]).animate_plot(res[1], res[2], save_file=True)

    comparison_animation(g_res[1], g_res[2], p_res[1], p_res[2], save_file=True)

    # plot goehring and par3add initial condition and polarised steady-state
    # together in two sub-figures
    plot_comparison_t0_tl(g_res[1], g_res[2], p_res[1], p_res[2])


def do_establishment_param_variation(Nx, init_conds, parameters_goehring,
                                     parameters_par3add, points_per_second,
                                     tL_est):
    print("Testing polarity establishment of goehring and par3add")
    filename_establishment_tasks = "comparison_par3addition_goehring_establishment_tasks_save"
    force_run = False  # option to force fresh run, don't use save
    # get tasks
    tasks_g, index_100x_g, xticks_g = get_model_variation_tasks(MODELS.GOEHRING,
                                                                Nx, tL_est,
                                                                init_conds[MODELS.GOEHRING],
                                                                parameters_goehring,
                                                                {key: parameters_goehring[key] for key in
                                                                 variation_params_goehring})
    tasks_p, index_100x_p, xticks_p = get_model_variation_tasks(MODELS.PAR3ADD,
                                                                Nx, tL_est,
                                                                init_conds[MODELS.PAR3ADD],
                                                                parameters_par3add,
                                                                {key: parameters_par3add[key] for key in
                                                                 variation_params_par3add})
    # get filenames
    filename_establishment_tasks_goehring = variation_task_helper.generate_variation_save_filename(
        filename_establishment_tasks,
        MODELS.GOEHRING, Nx, tL_est, points_per_second,
        parameters_goehring, variation_params_goehring,
        init_conds[MODELS.GOEHRING], tasks_g)
    filename_establishment_tasks_par3add = variation_task_helper.generate_variation_save_filename(
        filename_establishment_tasks,
        MODELS.PAR3ADD, Nx, tL_est, points_per_second,
        parameters_par3add, variation_params_par3add,
        init_conds[MODELS.PAR3ADD], tasks_p)

    print(f"attempting load of {filename_establishment_tasks_goehring} and {filename_establishment_tasks_par3add}")

    # try to load from savedata
    load_data_goehring = variation_task_helper.load_runs(filename_establishment_tasks_goehring)
    load_data_par3add = variation_task_helper.load_runs(filename_establishment_tasks_par3add)
    
    if (not force_run) and load_data_goehring[0] and load_data_par3add[0]:
        print("loading data succeeded")
        baseline_goehring = load_data_goehring[1]
        results_by_variable_goehring = load_data_goehring[2]
        baseline_par3add = load_data_par3add[1]
        results_by_variable_par3add = load_data_par3add[2]
    else:  # loading failed or partial failure
        print("loading failed")

        combined_tasks = []

        # check if some things loaded successfully
        # otherwise add to task queue
        if not load_data_goehring[0]:
            combined_tasks = combined_tasks + tasks_g
        else:
            baseline_goehring = load_data_goehring[1]
            results_by_variable_goehring = load_data_goehring[2]
            print("loaded goehring")

        if not load_data_par3add[0]:
            combined_tasks = combined_tasks + tasks_p
        else:
            baseline_par3add = load_data_par3add[1]
            results_by_variable_par3add = load_data_par3add[2]
            print("loaded par3add")


        # run tasks
        print(f"{len(combined_tasks)} sets of tasks to run, totalling {sum([len(combined_tasks[i]) for i in range(0, len(combined_tasks))])} tasks")

        res_out = variation_task_helper.run_grouped_tasks(combined_tasks)
        res_out_goehring = []
        res_out_par3add = []

        # sort results by model type
        for sub_task_arr in res_out:
            model_used = sub_task_arr[0][0]
            match model_used:
                case MODELS.GOEHRING:
                    res_out_goehring.append(sub_task_arr)
                case MODELS.PAR3ADD:
                    res_out_par3add.append(sub_task_arr)

        # save if it wasn't loaded from file
        if len(res_out_goehring) != 0:
            baseline_goehring, results_by_variable_goehring = variation_task_helper.split_baseline_from_results(
                MODELS.GOEHRING, res_out_goehring, index_100x_g)
            variation_task_helper.save_runs(filename_establishment_tasks_goehring, tasks_g,
                                            baseline_goehring, results_by_variable_goehring)
            print("Saved goehring runs")

        if len(res_out_par3add) != 0:
            baseline_par3add, results_by_variable_par3add = variation_task_helper.split_baseline_from_results(
                MODELS.PAR3ADD, res_out_par3add, index_100x_p)
            variation_task_helper.save_runs(filename_establishment_tasks_par3add, tasks_p,
                                            baseline_par3add, results_by_variable_par3add)
            print("Saved par3add runs")

    # Plot
    print("Plotting")

    # filenames
    fn1 = "establishment_variations_p1.png"
    fn2 = "establishment_variations_p2.png"

    Module_Goehring.plot_variation_sets(results_by_variable_goehring,
                                        label="goehring polarity establishment,tL=" + str(tL_est),
                                        x_axis_labels=xticks_g)
    plt.savefig(fn1, bbox_inches="tight")
    print(f"Saved goehring variation plot to {fn1}")
    Module_Par3Add.plot_variation_sets(results_by_variable_par3add,
                                       label="par3add polarity establishment,tL=" + str(tL_est),
                                       x_axis_labels=xticks_p)
    plt.savefig(fn2, bbox_inches="tight")
    print(f"Saved par3add variation plot to {fn2}")

    Module_Par3Add.animate_plot(baseline_par3add[0], baseline_par3add[1], save_file=True, file_code="baselinepar3addtestanimation")


def do_maintenance_param_variation(Nx, parameters_goehring, parameters_par3add,
                                   points_per_second, tL_est):
    print("Testing polarity maintenance of goehring and par3add")

    initial_conditions = {
        MODELS.GOEHRING: np.array([0]*(Nx//2) + [1.9]*(Nx-Nx//2)
                                  + [4.25]*(Nx//2) + [0]*(Nx-Nx//2)),
        MODELS.PAR3ADD: np.array([0]*(Nx//2) + [1.1]*(Nx-Nx//2)
                                 + [0]*(Nx//2) + [1.5]*(Nx-Nx//2)
                                 + [0]*(Nx//2) + [0.5]*(Nx-Nx//2)
                                 + [4.3]*(Nx//2) + [0]*(Nx-Nx//2)),
        }

    filename_maintenance_tasks = "comparison_par3addition_goehring_maintenance_tasks_save"
    # generate tasks
    tasks_g, index_100x_g, xticks_g = get_model_variation_tasks(MODELS.GOEHRING,
                                                                Nx, tL_est,
                                                                initial_conditions[MODELS.GOEHRING],
                                                                parameters_goehring,
                                                                {key: parameters_goehring[key] for key in
                                                                 variation_params_goehring})
    tasks_p, index_100x_p, xticks_p = get_model_variation_tasks(MODELS.PAR3ADD,
                                                                Nx, tL_est,
                                                                initial_conditions[MODELS.PAR3ADD],
                                                                parameters_par3add,
                                                                {key: parameters_par3add[key] for key in
                                                                 variation_params_par3add})
    # get filenames
    filename_maintenance_tasks_goehring = variation_task_helper.generate_variation_save_filename(
        filename_maintenance_tasks,
        MODELS.GOEHRING, Nx, tL_est, points_per_second,
        parameters_goehring, variation_params_goehring,
        initial_conditions[MODELS.GOEHRING], tasks_g)
    filename_maintenance_tasks_par3add = variation_task_helper.generate_variation_save_filename(
       filename_maintenance_tasks,
       MODELS.PAR3ADD, Nx, tL_est, points_per_second,
       parameters_par3add, variation_params_par3add,
       initial_conditions[MODELS.PAR3ADD], tasks_p)

    # try load from existing savedata
    print(f"attempting load of {filename_maintenance_tasks_goehring}\n \
           and {filename_maintenance_tasks_par3add}")
    load_data_goehring = variation_task_helper.load_runs(filename_maintenance_tasks_goehring)
    load_data_par3add = variation_task_helper.load_runs(filename_maintenance_tasks_par3add)

    if load_data_goehring[0] and load_data_par3add[0]:
        print("loading data succeeded")
        baseline_goehring = load_data_goehring[1]
        results_by_variable_goehring = load_data_goehring[2]
        baseline_par3add = load_data_par3add[1]
        results_by_variable_par3add = load_data_par3add[2]
    else:  # load failed or partially failed
        print("loading failed")

        combined_tasks = []

        # if failed to load, add to task list
        if not load_data_goehring[0]:
            combined_tasks = combined_tasks + tasks_g
        else:
            baseline_goehring = load_data_goehring[1]
            results_by_variable_goehring = load_data_goehring[2]
            print("loaded goehring")

        if not load_data_par3add[0]:
            combined_tasks = combined_tasks + tasks_p
        else:
            baseline_par3add = load_data_par3add[1]
            results_by_variable_par3add = load_data_par3add[2]
            print("loaded par3add")

        print(f"{len(combined_tasks)} sets of tasks to run, totalling {sum([len(combined_tasks[i]) for i in range(0, len(combined_tasks))])} tasks")

        # run tasks
        res_out = variation_task_helper.run_grouped_tasks(combined_tasks)
        res_out_goehring = []
        res_out_par3add = []

        # sort by model type
        for sub_task_arr in res_out:
            model_used = sub_task_arr[0][0]

            match model_used:
                case MODELS.GOEHRING:
                    res_out_goehring.append(sub_task_arr)
                case MODELS.PAR3ADD:
                    res_out_par3add.append(sub_task_arr)

        # if not loaded from savedata, save
        if len(res_out_goehring) != 0:
            baseline_goehring, results_by_variable_goehring = variation_task_helper.split_baseline_from_results(
                MODELS.GOEHRING, res_out_goehring, index_100x_g)
            variation_task_helper.save_runs(filename_maintenance_tasks_goehring,
                                            tasks_g, baseline_goehring,
                                            results_by_variable_goehring)
            print("Saved goehring runs")

        if len(res_out_par3add) != 0:
            baseline_par3add, results_by_variable_par3add = variation_task_helper.split_baseline_from_results(
                MODELS.PAR3ADD, res_out_par3add, index_100x_p)
            variation_task_helper.save_runs(filename_maintenance_tasks_par3add,
                                            tasks_p, baseline_par3add,
                                            results_by_variable_par3add)
            print("Saved par3add runs")

    # Plot
    print("Plotting")

    # filenames for figures
    fn1 = "maintenance_variations_p1.png"
    fn2 = "maintenance_variations_p2.png"

    Module_Goehring.plot_variation_sets(results_by_variable_goehring,
                                        label="goehring polarity maintenance,tL="+str(tL_est),
                                        x_axis_labels=xticks_p)
    plt.savefig(fn1, bbox_inches="tight")
    
    print(f"Saved goehring variation plot to {fn1}")
    Module_Par3Add.plot_variation_sets(results_by_variable_par3add,
                                       label="par3add polarity maintenance,tL="+str(tL_est),
                                       x_axis_labels=xticks_p)
    plt.savefig(fn2, bbox_inches="tight")
    
    print(f"Saved par3add variation plot to {fn2}")

    # Plot animation of baseline
    Module_Par3Add.animate_plot(baseline_par3add[0], baseline_par3add[1], save_file=True, file_code="baselinepar3addtestingmaintenance030325")
    Module_Goehring.animate_plot(baseline_goehring[0], baseline_goehring[1], save_file=True,file_code="baselinegoehringtestingmaintenance030325")


# animated plot of both goehring and par3add overlaying one another
def comparison_animation(pol_goehring_sol, pol_goehring_kvals, pol_par3add_sol, pol_par3add_kvals, file_code: str = None, save_file=False):
    if file_code is None:
        file_code = f'{time.time_ns()}'[5:]

    Nx_g = pol_goehring_kvals["Nx"]
    Nx_p = pol_par3add_kvals["Nx"]

    combined_apar = []

    for i in np.arange(0,len(pol_par3add_sol.t)):
        combined_apar.append(pol_par3add_sol.y[Nx_p:2*Nx_p,i] + pol_par3add_sol.y[2*Nx_p:3*Nx_p, i])

    fig, ax = plt.subplots()
    p_par3, = ax.plot(pol_par3add_kvals["X"], pol_par3add_sol.y[:Nx_p, 0], label="p_J", color="green")
    p_par6, = ax.plot(pol_par3add_kvals["X"], combined_apar[0], label="p_MA", color="purple")
    g_apar, = ax.plot(pol_goehring_kvals["X"], pol_goehring_sol.y[:Nx_g, 0], label="g_aPar", color="purple", linestyle="dashed")
    p_pPar, = ax.plot(pol_par3add_kvals["X"], pol_par3add_sol.y[3*Nx_p:, 0], label="p_pPar", color="orange")
    g_pPar, = ax.plot(pol_goehring_kvals["X"], pol_goehring_sol.y[Nx_g:, 0], label="g_pPar", color="orange", linestyle="dashed")

    p_polarity, _, _, = metric_functions.polarity_get_all(pol_par3add_kvals["X"],
                            combined_apar[0], pol_par3add_sol.y[3*Nx_p:,0],Nx_p)
    g_polarity, _, _, = metric_functions.polarity_get_all(pol_goehring_kvals["X"],
                            pol_goehring_sol.y[:Nx_g,0], pol_goehring_sol.y[Nx_g:,0],Nx_g)
    time_label = ax.text(0.2, 1.05, f"t={pol_par3add_sol.t[0]},{pol_goehring_sol.t[0]}; \
                            p={p_polarity:.2f},{g_polarity:.2f}", transform=ax.transAxes, ha="center")
    linev, = ax.plot(pol_par3add_kvals["X"], [pol_par3add_kvals["v_func"](pol_par3add_kvals, x, 0) for x in pol_par3add_kvals["X"]], label="v (from p)", linestyle="--", color="black")

    ax.text(0.7, 1.05, "goehring and par3add;Nx:" + str(Nx_p), transform=ax.transAxes, ha="center")

    maxy = np.max([np.max(pol_par3add_sol.y),np.max(pol_goehring_sol.y),np.max(combined_apar)])

    ax.set(xlim=[pol_par3add_kvals["x0"], pol_par3add_kvals["xL"]], ylim=[-0.05,maxy+0.05], xlabel="x", ylabel="par3,A/P")
    ax.legend()
    ax.set_title("compare goehring and par3add")
    
    def animate(t_i):
        p_par3.set_ydata(pol_par3add_sol.y[:Nx_p, t_i])
        p_par6.set_ydata(combined_apar[t_i])
        g_apar.set_ydata(pol_goehring_sol.y[:Nx_g, t_i])
        p_pPar.set_ydata(pol_par3add_sol.y[3*Nx_p:, t_i])
        g_pPar.set_ydata(pol_goehring_sol.y[Nx_g:, t_i])
        
        p_polarity, _, _, = metric_functions.polarity_get_all(pol_par3add_kvals["X"],
                                combined_apar[t_i], pol_par3add_sol.y[3*Nx_p:,t_i],Nx_p)
        g_polarity, _, _, = metric_functions.polarity_get_all(pol_goehring_kvals["X"],
                                pol_goehring_sol.y[:Nx_g,t_i], pol_goehring_sol.y[Nx_g:,t_i],Nx_g)
        time_label.set_text(f"t={pol_par3add_sol.t[t_i]},{pol_goehring_sol.t[t_i]}; \
                                p={p_polarity:.2f},{g_polarity:.2f}")
        
        linev.set_ydata([pol_par3add_kvals["v_func"](pol_par3add_kvals, x, t_i) for x in pol_par3add_kvals["X"]])
        
        return (p_par3,p_par6,g_apar,p_pPar,g_pPar,time_label,linev)

    ani = animation.FuncAnimation(fig, animate, interval=10000/len(pol_par3add_sol.t), blit=True, frames=len(pol_par3add_sol.t))

    if save_file:
        file_name = f"{file_code}_compare.mp4"
        print(f"Saving animation to {file_name}")
        ani.save(file_name)
    else:
        plt.show(block=True)


# plot two-panel figure comparing intiial condition
# and polarised state of goehring model and par3add model
def plot_comparison_t0_tl(pol_goehring_sol, pol_goehring_kvals, pol_par3add_sol, pol_par3add_kvals):
    Nx_g = pol_goehring_kvals["Nx"]
    Nx_p = pol_par3add_kvals["Nx"]

    combined_apar = []

    for i in np.arange(0,len(pol_par3add_sol.t)):
        combined_apar.append(pol_par3add_sol.y[Nx_p:2*Nx_p,i] + pol_par3add_sol.y[2*Nx_p:3*Nx_p, i])
    
    fig, axs = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

    # plot t=0
    ax_t0 = axs[0]
    ax_t0.plot(pol_par3add_kvals["X"], pol_par3add_sol.y[:Nx_p, 0],
                         label=figure_helper.par3add_labels[0],
                         color=figure_helper.par3add_colours[0],
                         linewidth=figure_helper.line_width,
                     )
    ax_t0.plot(pol_par3add_kvals["X"], combined_apar[0],
                         label="Par6",
                         color=figure_helper.par3add_colours[1],
                         linewidth=figure_helper.line_width,
                     )
    ax_t0.plot(pol_par3add_kvals["X"], pol_par3add_sol.y[3*Nx_p:, 0],
                         label="pPar",
                         color=figure_helper.par3add_colours[3],
                         linewidth=figure_helper.line_width,
                     )
    ax_t0.plot(pol_goehring_kvals["X"], pol_goehring_sol.y[:Nx_g, 0],
                         label="Par6 Goehring",
                         color=figure_helper.goehring_colours[0],
                         linewidth=figure_helper.line_width,
                         linestyle="dashed"
                     )
    ax_t0.plot(pol_goehring_kvals["X"], pol_goehring_sol.y[Nx_g:, 0],
                         label="pPar Goehring",
                         color=figure_helper.goehring_colours[1],
                         linewidth=figure_helper.line_width,
                         linestyle="dashed"
                     )
    
    

    ax_t0.tick_params(which="both", labelsize=figure_helper.font_size)
    ax_t0.set_title("t=0", fontsize=figure_helper.font_size)
    ax_t0.set_xlabel(figure_helper.xlabel, fontsize=figure_helper.font_size)
    ax_t0.set_ylabel(figure_helper.ylabel, fontsize=figure_helper.font_size)
    ax_t0.text(0.98, 0.95, f"New Model: p={metric_functions.polarity_measure(pol_par3add_kvals["X"],
                                                                           pol_par3add_sol.y[Nx_p:2*Nx_p, 0]+pol_par3add_sol.y[2*Nx_p:3*Nx_p, 0],
                                                                           pol_par3add_sol.y[3*Nx_p:, 0], Nx_p):.2f}",
               transform=ax_t0.transAxes, ha="right", fontsize=figure_helper.label_font_size)
    ax_t0.text(0.98, 0.91, f"Goehring: p={metric_functions.polarity_measure(pol_goehring_kvals["X"],
                                                                           pol_goehring_sol.y[:Nx_g, 0],
                                                                           pol_goehring_sol.y[Nx_g:, 0], Nx_g):.2f}",
               transform=ax_t0.transAxes, ha="right", fontsize=figure_helper.label_font_size)

    # plot t=tL
    ax_tL = axs[1]
    ax_tL.plot(pol_par3add_kvals["X"], pol_par3add_sol.y[:Nx_p, -1],
                         label=figure_helper.par3add_labels[0],
                         color=figure_helper.par3add_colours[0],
                         linewidth=figure_helper.line_width,
                     )
    ax_tL.plot(pol_par3add_kvals["X"], combined_apar[-1],
                         label="Par6",
                         color=figure_helper.par3add_colours[1],
                         linewidth=figure_helper.line_width,
                     )
    ax_tL.plot(pol_par3add_kvals["X"], pol_par3add_sol.y[3*Nx_p:, -1],
                         label="pPar",
                         color=figure_helper.par3add_colours[3],
                         linewidth=figure_helper.line_width,
                     )
    ax_tL.plot(pol_goehring_kvals["X"], pol_goehring_sol.y[:Nx_g, -1],
                         label="Par6 Goehring",
                         color=figure_helper.goehring_colours[0],
                         linewidth=figure_helper.line_width,
                         linestyle="dashed"
                     )
    ax_tL.plot(pol_goehring_kvals["X"], pol_goehring_sol.y[Nx_g:, -1],
                         label="pPar Goehring",
                         color=figure_helper.goehring_colours[1],
                         linewidth=figure_helper.line_width,
                         linestyle="dashed"
                     )
    
    
    ax_tL.tick_params(which="both", labelsize=figure_helper.font_size)
    ax_tL.set_title(f"t={pol_par3add_sol.t[-1]:.0f}", fontsize=figure_helper.font_size)
    ax_tL.set_xlabel(figure_helper.xlabel, fontsize=figure_helper.font_size)
    ax_tL.text(0.98, 0.95, f"New Model: p={metric_functions.polarity_measure(pol_par3add_kvals["X"],
                                                                           pol_par3add_sol.y[Nx_p:2*Nx_p, -1]+pol_par3add_sol.y[2*Nx_p:3*Nx_p, -1],
                                                                           pol_par3add_sol.y[3*Nx_p:, -1], Nx_p):.2f}",
               transform=ax_tL.transAxes, ha="right", fontsize=figure_helper.label_font_size)
    ax_tL.text(0.98, 0.91, f"Goehring: p={metric_functions.polarity_measure(pol_goehring_kvals["X"],
                                                                           pol_goehring_sol.y[:Nx_g, -1],
                                                                           pol_goehring_sol.y[Nx_g:, -1], Nx_g):.2f}",
               transform=ax_tL.transAxes, ha="right", fontsize=figure_helper.label_font_size)

    
    
    axs[0].legend(loc="upper left", fontsize=figure_helper.font_size, handlelength=4)
    plt.xticks([0, 70])
    plt.yticks([0, 1, 2, 3, 4])

    fig.set_size_inches(16,6)
    plt.savefig("emergence_compare_par3add_goehring_t0_tL.pdf", bbox_inches="tight")


if __name__ == '__main__':
    matplotlib.use('Agg')  # block plots from appearing
    main()
    plt.show()

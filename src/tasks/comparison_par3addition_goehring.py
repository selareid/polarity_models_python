# run par3addition model and generate plots for comparison to goehring

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
from matplotlib import pyplot as plt, animation
from src import model_task_handler
from src.tasks import variation_task_helper
from matplotlib import pyplot as plt
from ..models import MODELS, model_to_module, model_to_string, metric_functions
import numpy as np


def v_func_zero(kvals, x, t):
    return 0


Module_Par3Add = model_to_module(MODELS.PAR3ADD)
Module_Goehring = model_to_module(MODELS.GOEHRING)

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

    "D_J": params_goehring["D_A"],
    "D_M": params_goehring["D_P"]/2, # this should be less diffusive, 0 is bad tho
    "D_A": params_goehring["D_A"],
    "D_P": params_goehring["D_P"],

    "kJP": params_goehring["k_AP"]/2,
    "kMP": 0.07,
    "kAP": params_goehring["k_AP"]/2,
    "kPA": params_goehring["k_PA"],

    "konJ": 0.014,
    "konA": 0,  # not used in writeup
    "konP": params_goehring["k_onP"],

    "koffJ": params_goehring["k_offA"]/2,
    "koffM": params_goehring["k_offA"],
    "koffA": params_goehring["k_offA"],
    "koffP": params_goehring["k_offP"],

    "k1": params_goehring["k_onA"],
    "k2": 0.0027,
 
    "rho_J": 1.2,
    "rho_A": params_goehring["rho_A"],
    "rho_P": params_goehring["rho_P"],

    "sigmaJ": 1,"sigmaM": 1,"sigmaP": 1,
    "alpha": 1, "beta": 2,
   }


variation_params_par3add = [
    "psi", "D_J", "D_M", "D_A", "D_P", "sigmaJ", "sigmaM", "sigmaP",
    "k1", "k2", "kAP", "kJP", "kMP", "kPA", "alpha", "beta",
    "rho_A", "rho_J", "rho_P", "konA", "konJ", "konP", "koffA", "koffJ", "koffP"
]

variation_params_goehring = [
    "psi", "D_A", "D_P", "k_onA", "k_onP", "k_offA", "k_offP",
    "k_AP", "k_PA", "rho_A", "rho_P", "alpha", "beta"
]



# model-agnostic function
def get_model_variation_tasks(model: MODELS, Nx: int, tL: int, initial_condition, parameters: dict, varied_params: dict):
    variation_multipliers, index_for_100x = variation_task_helper.get_variation_multiplier()
    xticks = variation_task_helper.get_xticks(variation_multipliers, index_for_100x)

    
    tasks: list[list[tuple]] = variation_task_helper.generate_tasks(model, variation_multipliers, varied_params, {**parameters, "Nx":Nx, "tL":tL,"initial_condition":initial_condition})

    return tasks, index_for_100x, xticks


def main():
    Nx = 100
    # tL_ic
    tL_est = 9000
    points_per_second = 0.1

    # homogeneous_ic_par3add = [0] * Nx + [0] * Nx + [0] * Nx + [0] * Nx
    homogeneous_ic_par3add = [1] * Nx + [1] * Nx + [1] * Nx + [0] * Nx
    # homogeneous_ic_goehring = [0]*Nx + [0]*Nx
    # homogeneous_ic_goehring = [2.828]*Nx + [0.025]*Nx
    homogeneous_ic_goehring = [2.828] * Nx + [0] * Nx
    parameters_par3add = {**params_par3add,
        "points_per_second": points_per_second,
        "Nx": Nx,
    }
    parameters_goehring = {**params_goehring,
        "points_per_second": points_per_second,
        "Nx": Nx,
    }

    ## run without advection to get initial condition (IC)
    init_conds = get_homogeneous_initial_conditions(homogeneous_ic_goehring, homogeneous_ic_par3add,
                                                    parameters_goehring, parameters_par3add)

    ## Run with advection using found IC
    do_establishment_run(init_conds, parameters_goehring, parameters_par3add, tL_est)

    # Do emergence/establishment variation test
    # do_establishment_param_variation(Nx, init_conds, parameters_goehring, parameters_par3add, points_per_second, tL_est)

    # plt.show(block=True)

    # TODO, don't include in commit
    # on my linux install, stuff doesn't render correctly so I just auto-close it
    plt.close('all')


def get_homogeneous_initial_conditions(homogeneous_ic_goehring, homogeneous_ic_par3add, parameters_goehring,
                                       parameters_par3add):
    # TODO, use variables here for label and tL
    tasks_initial_condition = [(MODELS.PAR3ADD, {
        **parameters_par3add, "initial_condition": homogeneous_ic_par3add, "label": "par3add finding IC", "tL": 5000,
        "v_func": v_func_zero}),
                               (MODELS.GOEHRING, {
                                   **parameters_goehring, "initial_condition": homogeneous_ic_goehring,
                                   "label": "goehring finding IC", "tL": 5000, "v_func": v_func_zero})
                               ]
    multi_res_initial_condition = model_task_handler.run_tasks_parallel(tasks_initial_condition)  # [0]
    init_conds = {}
    for res in multi_res_initial_condition:
        print(res[0])
        # print("A" + res[1])
        match res[0]:
            case MODELS.GOEHRING:
                init_conds[MODELS.GOEHRING] = res[1].y[:, -1]
            case MODELS.PAR3ADD:
                init_conds[MODELS.PAR3ADD] = res[1].y[:, -1]

        # model_to_module(res[0]).animate_plot(res[1], res[2], save_file=True, file_code="hehehehehe"+str(res[0])+"11111")
    # TODO split goehring and par3add run to get initial condition for each
    print("Stats of homogeneous F.P.:")
    print(
        f"Par3add Model: par3: {init_conds[MODELS.PAR3ADD][0]}, par3-PKC: {init_conds[MODELS.PAR3ADD][parameters_par3add["Nx"]]}, \
        cdc42-PKC: {init_conds[MODELS.PAR3ADD][2*parameters_par3add["Nx"]]}, pPar: {init_conds[MODELS.PAR3ADD][len(init_conds[MODELS.PAR3ADD]) - 1]}")
    print(
        f"Goehring Model: aPar: {init_conds[MODELS.GOEHRING][0]}, pPar: {init_conds[MODELS.GOEHRING][len(init_conds[MODELS.GOEHRING]) - 1]}")
    
    # plt.show(block=True)

    return init_conds


def do_establishment_run(init_conds, parameters_goehring, parameters_par3add, tL_est):
    tasks_establishment = [(MODELS.PAR3ADD, {
        **parameters_par3add, "label": "par3add establishment", "tL": tL_est,
        "initial_condition": init_conds[MODELS.PAR3ADD]}),
                           (MODELS.GOEHRING, {
                               **parameters_goehring, "label": "goehring establishment", "tL": tL_est,
                               "initial_condition": init_conds[MODELS.GOEHRING]})
                           # "initial_condition": homogeneous_ic_goehring})
                           ]
    multi_res_establishment = model_task_handler.run_tasks_parallel(tasks_establishment)  # [0]
    for res in multi_res_establishment:
        match res[0]:
            case MODELS.GOEHRING:
                #     Module_Goehring.animate_plot

                #     # plot_overall_quantities_over_time
                #     # plot_metric_comparisons

                #     # panic
                # pass  # TODO
                g_res = res
            case MODELS.PAR3ADD:
                # Module_Par3Add.animate_plot_apar_combo(res[1], res[2], save_file=True, no_par3=True)
                # Module_Par3Add.animate_plot_apar_combo(res[1], res[2], save_file=True, no_par3=False)
                p_res = res

            #     Module_Par3Add.animate_plot(res[1], res[2], save_file=True)

        # model_to_module(res[0]).animate_plot(res[1], res[2], save_file=True)
    
    comparison_animation(g_res[1], g_res[2], p_res[1], p_res[2], save_file=True)


def do_establishment_param_variation(Nx, init_conds, parameters_goehring, parameters_par3add, points_per_second,
                                     tL_est):
    print(f"Testing polarity establishment of goehring and par3add")
    filename_establishment_tasks = "comparison_par3addition_goehring_establishment_tasks_save"
    force_run = False
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

    print("attempting load")
    load_data_goehring = variation_task_helper.load_runs(filename_establishment_tasks_goehring)
    load_data_par3add = variation_task_helper.load_runs(filename_establishment_tasks_par3add)
    if (not force_run) and load_data_goehring[0] and load_data_par3add[0]:
        print("loading data succeeded")
        baseline_goehring = load_data_goehring[1]
        results_by_variable_goehring = load_data_goehring[2]
        baseline_par3add = load_data_par3add[1]
        results_by_variable_par3add = load_data_par3add[2]
    else:
        print("loading failed")

        combined_tasks = []

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
            print("loading par3add")

        print(f"{len(combined_tasks)} sets of tasks to run, totalling {sum([len(combined_tasks[i]) for i in range(0, len(combined_tasks))])} tasks")

        res_out = variation_task_helper.run_grouped_tasks(combined_tasks, 11)
        res_out_goehring = []
        res_out_par3add = []

        for sub_task_arr in res_out:
            model_used = sub_task_arr[0][0]
            # print(model_to_string(model_used))
            # pass
            match model_used:
                case MODELS.GOEHRING:
                    res_out_goehring.append(sub_task_arr)
                case MODELS.PAR3ADD:
                    res_out_par3add.append(sub_task_arr)

        if len(res_out_goehring) == 0:
            baseline_goehring, results_by_variable_goehring = variation_task_helper.split_baseline_from_results(
                MODELS.GOEHRING, res_out_goehring, index_100x_g)
        if len(res_out_par3add) == 0:
            baseline_par3add, results_by_variable_par3add = variation_task_helper.split_baseline_from_results(
                MODELS.PAR3ADD, res_out_par3add, index_100x_p)

        variation_task_helper.save_runs(filename_establishment_tasks_goehring, tasks_g,
                                        baseline_goehring, results_by_variable_goehring)
        variation_task_helper.save_runs(filename_establishment_tasks_par3add, tasks_p,
                                        baseline_par3add, results_by_variable_par3add)
        
    # Plot
    print(f"Plotting")
    Module_Goehring.plot_variation_sets(results_by_variable_goehring,
                                        label="goehring polarity establishment,tL=" + str(tL_est),
                                        x_axis_labels=xticks_g)
    plt.savefig(f"130225_p1.png")
    Module_Par3Add.plot_variation_sets(results_by_variable_par3add,
                                       label="par3add polarity establishment,tL=" + str(tL_est),
                                       x_axis_labels=xticks_p)
    plt.savefig(f"130225_p2.png")

    Module_Par3Add.animate_plot(baseline_par3add[0], baseline_par3add[1], save_file=True, file_code="baselinepar3addtestingahhh140215")


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
        file_name = f"{file_code}_spatialPar_compare.mp4"
        print(f"Saving animation to {file_name}")
        ani.save(file_name)
    else:
        plt.show(block=True)


if __name__ == '__main__':
    main()
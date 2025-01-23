# run par3addition model and generate plots for comparison to goehring

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src import model_task_handler
from src.tasks import variation_task_helper
from matplotlib import pyplot as plt
from ..models import MODELS, model_to_module, model_to_string


def v_func_zero(kvals, x, t):
    return 0


Module_Par3Add = model_to_module(MODELS.PAR3ADD)
Module_Goehring = model_to_module(MODELS.GOEHRING)

# params_par3add = {
#     "psi": 0.174,
#     "D_J": 0.28,"D_M": 0.28,"D_A": 0.28,"D_P": 0.15,
#     "sigmaJ": 1,"sigmaM": 1,"sigmaP": 1,
#     "k1": 8.58*10**(-3), "k2": 7.5*10**(-3),
#     "kAP": 0.19, "kJP": 0.0010, "kMP": 0.006, "kPA": 0.1196,
#     "alpha": 1, "beta": 2,
#     "rho_A": 1.56, "rho_J": 0.7, "rho_P": 1,
#     "konA": 0, "konJ": 0.00858, "konP": 0.0474,
#     "koffA": 0.00545, "koffJ": 0.001, "koffP": 0.008,
# }

# params_par3add = {
#     "psi": 0.174,
#     "D_J": 0.28,"D_M": 0.28,"D_A": 0.28,"D_P": 0.15,
#     "sigmaJ": 1,"sigmaM": 1,"sigmaP": 1,
#     "k1": 8.58*10**(-3), "k2": 7.5*10**(-3),
#     "kAP": 0.19, "kJP": 0.0037, "kMP": 0.006, "kPA": 0.1196,
#     "alpha": 1, "beta": 2,
#     "rho_A": 1.323, "rho_J": 1.125, "rho_P": 1,
#     "konA": 0, "konJ": 0.00858, "konP": 0.0474,
#     "koffA": 0.0054, "koffJ": 0.0001, "koffP": 0.0073,
# }

params_par3add = {
    "psi": 0.174,
    "D_J": 0.28,"D_M": 0.28,"D_A": 0.28,"D_P": 0.15,
    "sigmaJ": 1,"sigmaM": 1,"sigmaP": 1,
    "k1": 8.58*10**(-3), "k2": 7.5*10**(-3),
    "kAP": 0.19, "kJP": 0.0037, "kMP": 0.006, "kPA": 0.1196,
    "alpha": 1, "beta": 2,
    "rho_A": 1.323, "rho_J": 0.6, "rho_P": 1,
    "konA": 0, "konJ": 0.00858, "konP": 0.0474,
    "koffA": 0.0054, "koffJ": 0.0001, "koffP": 0.0073,
}

params_goehring = {
    "psi": 0.174,
    "D_A": 0.28, "D_P": 0.15,
    "k_onA": 8.58 * 10 ** (-3), "k_onP": 4.74 * 10 ** (-2),
    "k_offA": 5.4 * 10 ** (-3), "k_offP": 7.3 * 10 ** (-3),
    "k_AP": 0.190, "k_PA": 2.0,
    "rho_A": 1.56, "rho_P": 1.0,
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
    model_module = model_to_module(model)

    variation_multipliers, index_for_100x = variation_task_helper.get_variation_multiplier()
    xticks = variation_task_helper.get_xticks(variation_multipliers, index_for_100x)

    
    tasks: list[list[tuple]] = variation_task_helper.generate_tasks(model, variation_multipliers, varied_params, parameters)

    return tasks, index_for_100x, xticks


def main():
    Nx = 100
    # tL_ic
    tL_est = 9000
    points_per_second = 0.1

    homogeneous_ic_par3add = [0] * Nx + [0] * Nx + [0] * Nx + [0] * Nx
    # homogeneous_ic_goehring = [0]*Nx + [0]*Nx
    # homogeneous_ic_goehring = [2.828]*Nx + [0.025]*Nx
    homogeneous_ic_goehring = [2.828] * Nx + [0] * Nx
    parameters_par3add = {
        # TODO
        **params_par3add,
        "points_per_second": points_per_second,
        "Nx": Nx,
    }
    parameters_goehring = {
        # TODO
        **params_goehring,
        "points_per_second": points_per_second,
        "Nx": Nx,
    }

    ## run without advection to get initial condition (IC)
    init_conds = get_homogeneous_initial_conditions(homogeneous_ic_goehring, homogeneous_ic_par3add,
                                                    parameters_goehring, parameters_par3add)

    # assert (init_conds[MODELS.PAR3ADD][0] > init_conds[MODELS.PAR3ADD][len(init_conds[MODELS.PAR3ADD]) - 1])
    # assert(init_conds[MODELS.GOEHRING][0] > init_conds[MODELS.GOEHRING][len(init_conds[MODELS.GOEHRING])-1]) TODO CHECK

    ## Run with advection using found IC
    # do_establishment_run(init_conds, parameters_goehring, parameters_par3add, tL_est)

    # Do emergence/establishment variation test
    do_establishment_param_variation(Nx, init_conds, parameters_goehring, parameters_par3add, points_per_second, tL_est)

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
                pass  # TODO
            case MODELS.PAR3ADD:
                Module_Par3Add.animate_plot_apar_combo(res[1], res[2], save_file=True, no_par3=True)
                Module_Par3Add.animate_plot_apar_combo(res[1], res[2], save_file=True, no_par3=False)

            #     Module_Par3Add.animate_plot(res[1], res[2], save_file=True)

        model_to_module(res[0]).animate_plot(res[1], res[2], save_file=True)


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
    # filename_establishment_tasks + "_" + model_to_string(MODELS.GOEHRING)
    filename_establishment_tasks_par3add = variation_task_helper.generate_variation_save_filename(
        filename_establishment_tasks,
        MODELS.PAR3ADD, Nx, tL_est, points_per_second,
        parameters_par3add, variation_params_par3add,
        init_conds[MODELS.PAR3ADD], tasks_p)
    # filename_establishment_tasks + "_" + model_to_string(MODELS.PAR3ADD)
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
        combined_tasks = combined_tasks + tasks_g
        combined_tasks = combined_tasks + tasks_p

        print(
            f"{len(combined_tasks)} sets of tasks to run, totalling {sum([len(combined_tasks[i]) for i in range(0, len(combined_tasks))])} tasks")

        res_out = variation_task_helper.run_grouped_tasks(combined_tasks)
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

        baseline_goehring, results_by_variable_goehring = variation_task_helper.split_baseline_from_results(
            MODELS.GOEHRING,
            res_out_goehring, index_100x_g)
        baseline_par3add, results_by_variable_par3add = variation_task_helper.split_baseline_from_results(
            MODELS.PAR3ADD,
            res_out_par3add, index_100x_p)

        variation_task_helper.save_runs(filename_establishment_tasks_goehring, tasks_g,
                                        baseline_goehring, results_by_variable_goehring)
        variation_task_helper.save_runs(filename_establishment_tasks_par3add, tasks_p,
                                        baseline_par3add, results_by_variable_par3add)
    # Plot
    print(f"Plotting")
    Module_Goehring.plot_variation_sets(results_by_variable_goehring,
                                        label="goehring polarity establishment,tL=" + str(tL_est),
                                        x_axis_labels=xticks_g)
    Module_Par3Add.plot_variation_sets(results_by_variable_par3add,
                                       label="par3add polarity establishment,tL=" + str(tL_est),
                                       x_axis_labels=xticks_p)
    # model_module.plot_variation_sets(results_by_variable, label=""+model_to_string(model), x_axis_labels=xticks)


if __name__ == '__main__':
    main()
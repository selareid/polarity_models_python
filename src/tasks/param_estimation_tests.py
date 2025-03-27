# Some parameter estimation tests
# These were used to inform initial conditions
# for the grid search that we used
# to get the chosen par3add parameter set

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src import model_task_handler
import matplotlib
from matplotlib import pyplot as plt
from ..models import MODELS, model_to_module
import copy


def v_func_zero(kvals, x, t):
    return 0


def quick_animate(file_code, save_plot, res):
    if res[1] != "FAILURE":
        model_to_module(res[0]).animate_plot(res[1], res[2], save_file=save_plot, file_code=file_code)
    else:
        print("Ahh Run Failed")


def quick_run_animate(file_code, save_plot, tasks):
    return model_task_handler.run_tasks_parallel(tasks,
                                                 min(len(tasks), 10),
                                                 lambda res: quick_animate(f"{file_code}_{res[2]["label"]}", save_plot, res))


MODULE_GOEHRING = model_to_module(MODELS.GOEHRING)
MODULE_PAR3ADD = model_to_module(MODELS.PAR3ADD)

Nx = 100

base_params_goehring = {
    "psi": 0.174,
    "D_A": 0.28, "D_P": 0.15,
    "k_onA": 8.58 * 10 ** (-3), "k_onP": 4.74 * 10 ** (-2),
    "k_offA": 5.4 * 10 ** (-3), "k_offP": 7.3 * 10 ** (-3),
    "k_AP": 0.190, "k_PA": 2.0,
    "rho_A": 1.56, "rho_P": 1.0,
    "alpha": 1, "beta": 2,

    "Nx": Nx, "points_per_second": 0.1, "tL": 3000,
    "initial_condition": [1]*Nx + [0]*Nx,
    "v_func": v_func_zero,
}


def do_goehring_steady_state(save_plot=True):
    tasks = [
            (MODELS.GOEHRING, {**base_params_goehring, "label": "base"}),
            (MODELS.GOEHRING, {**base_params_goehring, "label": "no_p", "k_onP": 0}),
            (MODELS.GOEHRING, {**base_params_goehring, "label": "base_0init", "initial_condition": [0]*(2*Nx)})
        ]

    all_res = model_task_handler.run_tasks_parallel(tasks, 2)

    for res in all_res:
        MODULE_GOEHRING.animate_plot(res[1], res[2], save_plot, "parm_est_via_goehring_"+res[2]["label"])


def do_par3add_J_M_only(save_plot=True):
    # Initial Test That Worked
    init_cond = {"J": 0, "M": 0, "A": 0, "P": 0}
    params_par3addition = {
        "psi": 0.174,

        "D_J": 0,
        "D_M": 0,
        "D_A": 0,
        "D_P": 0,

        "kJP": 0,
        "kMP": 0,
        "kAP": 0,
        "kPA": 0,


        "konJ": base_params_goehring["k_onA"],
        "konA": 0,  # variable not used in writeup
        "konP": 0,

        "koffJ": base_params_goehring["k_offA"],
        "koffM": base_params_goehring["k_offA"],
        "koffA": base_params_goehring["k_offA"],
        "koffP": 0,

        "k1": base_params_goehring["k_onA"],
        "k2": 0,

        "rho_J": 1.7,
        "rho_A": base_params_goehring["rho_A"],
        "rho_P": 0,

        "v_func": v_func_zero,
        "Nx": Nx, "tL": 3000,
        "points_per_second": 0.1,
        "initial_condition":
            [init_cond["J"]]*Nx +
            [init_cond["M"]]*Nx +
            [init_cond["A"]]*Nx +
            [init_cond["P"]]*Nx,
    }

    tasks = [
            (MODELS.PAR3ADD, {**params_par3addition})
        ]

    all_res = model_task_handler.run_tasks(tasks)

    for res in all_res:
        if res[1] != "FAILURE":
            MODULE_PAR3ADD.animate_plot(res[1], res[2], save_file=save_plot, file_code=f"par3add_run_noA_310125code_{res[2]["label"]}")
        else:
            print("Ahh Run Failed")


def do_par3add_J_M_A(save_plot=True):
    init_cond = {"J": 0, "M": 0, "A": 0, "P": 0}

    params_par3addition = {
        "psi": 0.174,

        "D_J": 0,
        "D_M": 0,
        "D_A": 0,
        "D_P": 0,

        "kJP": 0,
        "kMP": 0,
        "kAP": 0,
        "kPA": 0,

        "konJ": base_params_goehring["k_onA"] / 2,
        "konA": 0,  # not used in writeup
        "konP": 0,

        "koffJ": base_params_goehring["k_offA"],
        "koffM": base_params_goehring["k_offA"],
        "koffA": base_params_goehring["k_offA"],
        "koffP": 0,

        "k1": base_params_goehring["k_onA"],
        "k2": base_params_goehring["k_onA"],

        "rho_J": 1.2,
        "rho_A": base_params_goehring["rho_A"],
        "rho_P": 0,

        "v_func": v_func_zero,
        "Nx": Nx, "tL": 3000,
        "points_per_second": 0.1,
        "initial_condition":
            [init_cond["J"]]*Nx +
            [init_cond["M"]]*Nx +
            [init_cond["A"]]*Nx +
            [init_cond["P"]]*Nx,
    }

    tasks = [
            (MODELS.PAR3ADD, {**params_par3addition})
        ]

    all_res = model_task_handler.run_tasks(tasks)

    for res in all_res:
        if res[1] != "FAILURE":
            MODULE_PAR3ADD.animate_plot(res[1], res[2], save_file=True, file_code=f"par3add_run_040225code_{res[2]["label"]}")
        else:
            print("Ahh Run Failed")


def do_par3add_J_M_A_P(save_plot=True):
    # here we have two runs, different initial conditions
    params_par3addition = {
        "psi": 0.174,

        "D_J": 0,
        "D_M": 0,
        "D_A": 0,
        "D_P": 0,

        "kJP": base_params_goehring["k_AP"]/3,
        "kMP": base_params_goehring["k_AP"]/3,
        "kAP": base_params_goehring["k_AP"]/3,
        "kPA": base_params_goehring["k_PA"],

        "konJ": base_params_goehring["k_onA"],
        "konA": 0,  # not used in writeup
        "konP": base_params_goehring["k_onP"],

        "koffJ": base_params_goehring["k_offA"],
        "koffM": base_params_goehring["k_offA"],
        "koffA": base_params_goehring["k_offA"],
        "koffP": base_params_goehring["k_offP"],

        "k1": base_params_goehring["k_onA"],
        "k2": base_params_goehring["k_onA"],

        "rho_J": 1.2,
        "rho_A": base_params_goehring["rho_A"],
        "rho_P": base_params_goehring["rho_P"],

        "v_func": v_func_zero,
        "Nx": Nx, "tL": 3000,
        "points_per_second": 0.1,
    }

    initial_condition_asymmetric = [0]*(Nx//2) + [1]*(Nx//2) \
        + [0]*(Nx//2) + [1]*(Nx//2) \
        + [0]*(Nx//2) + [1]*(Nx//2) \
        + [1]*(Nx//2) + [0]*(Nx//2)
    gap = 20
    initial_condition_asymmetric_w_gap = [0]*(Nx//2 + gap) + [1]*(Nx//2 - gap) \
        + [0]*(Nx//2 + gap) + [1]*(Nx//2 - gap) \
        + [0]*(Nx//2 + gap) + [1]*(Nx//2 - gap) \
        + [1]*(Nx//2 - gap) + [0]*(Nx//2 + gap)

    tasks = [
        (MODELS.PAR3ADD, {**params_par3addition, "label": "0_init", "initial_condition": [0]*(4*Nx)}),
        (MODELS.PAR3ADD, {**params_par3addition, "label": "A_up_init", "initial_condition": [0]*Nx + [1]*(2*Nx) + [0]*Nx}),
        (MODELS.PAR3ADD, {**params_par3addition, "label": "asym_init", "initial_condition": initial_condition_asymmetric}),
        (MODELS.PAR3ADD, {**params_par3addition, "label": "asym_init_w_gap", "initial_condition": initial_condition_asymmetric_w_gap}),
        ]

    _all_res = model_task_handler.run_tasks_parallel(tasks, max(len(tasks), 6), lambda res: quick_animate(f"par3add_run_060225code_{res[2]["label"]}", save_plot, res))


def do_par3add_spatial(save_plot=True):
    params_par3addition = {
        "psi": 0.174,

        "D_J": base_params_goehring["D_A"],
        "D_M": base_params_goehring["D_P"]/2,  # this should be less diffusive than D_A, 0 is bad tho
        "D_A": base_params_goehring["D_A"],
        "D_P": base_params_goehring["D_P"],

        "kJP": 0.08,
        "kMP": 0.07,
        "kAP": base_params_goehring["k_AP"]/2,
        "kPA": base_params_goehring["k_PA"],

        "konJ": base_params_goehring["k_onA"],
        "konA": 0,  # not used in writeup
        "konP": base_params_goehring["k_onP"],

        "koffJ": base_params_goehring["k_offA"]/2,
        "koffM": base_params_goehring["k_offA"],
        "koffA": base_params_goehring["k_offA"],
        "koffP": base_params_goehring["k_offP"],

        "k1": base_params_goehring["k_onA"],
        "k2": 0.0022,

        "rho_J": 1.2,
        "rho_A": base_params_goehring["rho_A"],
        "rho_P": base_params_goehring["rho_P"],

        "Nx": Nx, "tL": 9000,
        "points_per_second": 0.01,
        "initial_condition": [1]*(3*Nx) + [0]*Nx,
        }

    tasks = [
        (MODELS.PAR3ADD, {**params_par3addition, "label": "spatial_init_b1"}),
        ]

    # vary konJ a bit
    for i in range(1, 7):
        t = copy.deepcopy(tasks[0])

        t[1]["konJ"] = t[1]["konJ"]*(1-(i)/15)
        t[1]["label"] = f"{t[1]["label"]}_konJ{t[1]["konJ"]}"

        tasks.append(t)

    print([t[1]["label"] for t in tasks])

    _all_res = quick_run_animate("par3add_run_190225code", save_plot, tasks)


def main():
    matplotlib.use('Agg')

    do_goehring_steady_state()
    do_par3add_J_M_only()
    do_par3add_J_M_A()
    do_par3add_J_M_A_P()
    do_par3add_spatial()

    plt.close("all")


if __name__ == "__main__":
    main()

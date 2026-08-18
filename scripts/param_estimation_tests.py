# Some parameter estimation tests
# These were used to inform initial conditions
# for the grid search that we used
# to get the chosen par3add parameter set

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import matplotlib
from matplotlib import pyplot as plt
import copy

from polarity.utilities import model_task_handler, figure_helper as fh
from polarity.model_enums import MODELS, model_to_module


def v_func_zero(kvals, x, t):
    return 0



MODULE_GOEHRING = model_to_module(MODELS.GOEHRING)
MODULE_PAR3ADD = model_to_module(MODELS.PAR3ADD)

Nx = 100
update_parms = {
    "Nx": Nx,
    "tL": 3000,
    "points_per_second": 0.1
}

output_dir = fh.FIGURES_DIR / "param_est_tests"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



def do_goehring_steady_state():
    tasks = [
            (MODELS.GOEHRING, {"label": "base", **update_parms}),
            (MODELS.GOEHRING, {"label": "no_p", **update_parms, "konP": 0}),
            (MODELS.GOEHRING, {"label": "base_0init", **update_parms, "initial_condition": [0]*(2*Nx)})
        ]
    goehring_res_list = model_task_handler.run_tasks_parallel(tasks, 3)
    for label, res in goehring_res_list:
        fh.animate_plot(res[0], res[1], save_file = output_dir / f"goehring_{label}")


def do_par3add_J_M_only():
    # Initial Test That Worked
    params_par3addition = {
        **update_parms,
        "D_J": 0, "D_M": 0, "D_A": 0, "D_P": 0, # No Diffusion
        "kJP": 0, "kMP": 0, "kAP": 0, "kPA": 0, # No Antagonism

        "konJ": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,
        "konM": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,
        "koffJ": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffM": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffA": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,

        "rho_J": 1.7,
        "rho_A": MODULE_GOEHRING.DEFAULT_PARAMETERS.rho_A,

        "konP": 0,
        "kdisp": 0,
        "koffP": 0,
        "rho_P": 0,

        "v_func": v_func_zero,
        "initial_condition":[0]*(4*Nx)
    }

    sol, kvals = MODULE_PAR3ADD.run_model(params_par3addition)
    fh.animate_plot(sol, kvals, save_file = output_dir / f"par3add_run_noA")



def do_par3add_J_M_A():
    params_par3addition = {
        **update_parms,
        "D_J": 0, "D_M": 0, "D_A": 0, "D_P": 0, # No Diffusion
        "kJP": 0, "kMP": 0, "kAP": 0, "kPA": 0, # No Antagonism

        "konJ": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA / 2,
        "konP": 0,

        "koffJ": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffM": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffA": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffP": 0,

        "konM": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,
        "kdisp": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,

        "rho_J": 1.2,
        "rho_A": MODULE_GOEHRING.DEFAULT_PARAMETERS.rho_A,
        "rho_P": 0,

        "v_func": v_func_zero,
        "initial_condition": [0]*(4*Nx)
    }
    sol, kvals = MODULE_PAR3ADD.run_model(params_par3addition)
    fh.animate_plot(sol, kvals, save_file = output_dir / f"par3add_run_noP")



def do_par3add_J_M_A_P(save_plot=True):
    # here we have two runs, different initial conditions
    params_par3addition = {
        **update_parms,
        "D_J": 0, "D_M": 0, "D_A": 0, "D_P": 0, # No Diffusion

        "kJP": MODULE_GOEHRING.DEFAULT_PARAMETERS.kAP,
        "kMP": MODULE_GOEHRING.DEFAULT_PARAMETERS.kAP,
        "kAP": MODULE_GOEHRING.DEFAULT_PARAMETERS.kAP,
        "kPA": MODULE_GOEHRING.DEFAULT_PARAMETERS.kPA,

        "konJ": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,
        "konP": MODULE_GOEHRING.DEFAULT_PARAMETERS.konP,
        "konM": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,

        "koffJ": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffM": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffA": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffP": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffP,
        "kdisp": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,

        "rho_J": 1.2,
        "rho_A": MODULE_GOEHRING.DEFAULT_PARAMETERS.rho_A,
        "rho_P": MODULE_GOEHRING.DEFAULT_PARAMETERS.rho_P,

        "v_func": v_func_zero
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

    par3add_res_list = model_task_handler.run_tasks_parallel(tasks, 4)
    for label, res in par3add_res_list:
        fh.animate_plot(res[0], res[1], save_file = output_dir / f"par3add_{label}")


def do_par3add_spatial():
    params_par3addition = {
        **update_parms,

        "D_J": MODULE_GOEHRING.DEFAULT_PARAMETERS.D_A,
        "D_M": MODULE_GOEHRING.DEFAULT_PARAMETERS.D_P/2,  # this should be less diffusive than D_A, 0 is bad
        "D_A": MODULE_GOEHRING.DEFAULT_PARAMETERS.D_A,
        "D_P": MODULE_GOEHRING.DEFAULT_PARAMETERS.D_P,

        "kJP": 0.08,
        "kMP": 0.07,
        "kAP": MODULE_GOEHRING.DEFAULT_PARAMETERS.kAP,
        "kPA": MODULE_GOEHRING.DEFAULT_PARAMETERS.kPA,

        "konJ": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,
        "konP": MODULE_GOEHRING.DEFAULT_PARAMETERS.konP,

        "koffJ": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffM": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffA": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffA,
        "koffP": MODULE_GOEHRING.DEFAULT_PARAMETERS.koffP,

        "konM": MODULE_GOEHRING.DEFAULT_PARAMETERS.konA,
        "kdisp": 0.0022,

        "rho_J": 1.2,
        "rho_A": MODULE_GOEHRING.DEFAULT_PARAMETERS.rho_A,
        "rho_P": MODULE_GOEHRING.DEFAULT_PARAMETERS.rho_P,

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

    konJ_res_list = model_task_handler.run_tasks_parallel(tasks, 7)
    for label, res in konJ_res_list:
        fh.animate_plot(res[0], res[1], save_file = output_dir / f"par3add_{label}")


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

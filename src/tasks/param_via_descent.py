import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src import model_task_handler
from src.tasks import variation_task_helper
from matplotlib import pyplot as plt
from ..models import MODELS, model_to_module, model_to_string
from scipy import optimize
import numpy as np
from numpy import linalg

def v_func_zero(kvals, x, t):
    return 0

def quick_animate(file_code, save_plot, res):
    if res[1] != "FAILURE":
        model_to_module(res[0]).animate_plot(res[1], res[2], save_file=save_plot, file_code=file_code)
    else:
        print("Ahh Run Failed")

MODULE_GOEHRING = model_to_module(MODELS.GOEHRING)
MODULE_PAR3ADD = model_to_module(MODELS.PAR3ADD)

Nx = 100
tL_homo = 3000
tL_polarise = 9000

FAILURE_OUT = Nx*100


params_goehring = {
    "psi": 0.174,
    "D_A": 0.28, "D_P": 0.15,
    "k_onA": 8.58 * 10 ** (-3), "k_onP": 4.74 * 10 ** (-2),
    "k_offA": 5.4 * 10 ** (-3), "k_offP": 7.3 * 10 ** (-3),
    "k_AP": 0.190, "k_PA": 2.0,
    "rho_A": 1.56, "rho_P": 1.0,
    "alpha": 1, "beta": 2,

    "Nx": Nx, "points_per_second": 0.1,
}

params_par3add = {
    "psi": 0.174,
    
    "D_J": params_goehring["D_A"],
    "D_M": params_goehring["D_P"]/2, # this should be less diffusive, 0 is bad tho
    "D_A": params_goehring["D_A"],
    "D_P": params_goehring["D_P"],
    
    "kJP": params_goehring["k_AP"]/2,
    "kMP": params_goehring["k_AP"]/2,
    "kAP": params_goehring["k_AP"]/2,
    "kPA": params_goehring["k_PA"],

    "konJ": params_goehring["k_onA"],
    "konA": 0, # not used in writeup
    "konP": params_goehring["k_onP"],
    
    "koffJ": params_goehring["k_offA"]/2,
    "koffM": params_goehring["k_offA"],
    "koffA": params_goehring["k_offA"],
    "koffP": params_goehring["k_offP"],

    "k1": params_goehring["k_onA"],
    "k2": params_goehring["k_onA"],

    "rho_J": 1.282,
    "rho_A": params_goehring["rho_A"],
    "rho_P": params_goehring["rho_P"],
    
    # "v_func": v_func_zero, # use default v_func_zero
    "Nx": Nx, "tL": 9000,
    "points_per_second": 0.01,
    "initial_condition": [1]*Nx + [1]*(2*Nx) + [0]*Nx,
}


def main():
    # run goehring to get base
    goehring_homo_ic = get_goehring_homo_ic()
    goehring_polarised = get_goehring_polarised(goehring_homo_ic)

    args = {"varied_keys": ["kJP", "kMP", "kAP", "konJ", "koffJ", "k1", "k2", "rho_J"],
            "g_homo": goehring_homo_ic,
            "g_polarised": goehring_polarised,
        }
    
    f0 = lambda x: func(x, args)
    x0 = [params_par3add[k] for k in args["varied_keys"]]
    bounds = [(params_par3add[k]/100,params_par3add[k]*100) for k in args["varied_keys"]]
    res = optimize.minimize(f0, x0, bounds=bounds, tol=0.1, options={"maxiter": 8, "disp": True})

    print("Finished")
    print(res)
    print(res["x"])



def get_goehring_homo_ic():
    task = (MODELS.GOEHRING, {**params_goehring, "tL": tL_homo,
                                "initial_condition": [1]*Nx + [0]*Nx, "v_func": v_func_zero,})
    res = model_task_handler.run_tasks([task])[0]
    init_cond = res[1].y[:,-1]
    return init_cond

def get_goehring_polarised(initial_condition):
    task = (MODELS.GOEHRING, {**params_goehring, "tL": tL_polarise,
                                "initial_condition": initial_condition})
    res = model_task_handler.run_tasks([task])[0]
    init_cond = res[1].y[:,-1]
    return init_cond

def func(x, args):
    x_as_dict = dict(zip(args["varied_keys"], x))

    print(x_as_dict)

    task_ic = (MODELS.PAR3ADD, {**params_par3add, **x_as_dict, "tL": tL_homo,
                                "initial_condition": [1] * Nx + [1] * Nx + [1] * Nx + [0] * Nx})
    res1 = model_task_handler.run_tasks([task_ic])[0]

    if res1[1] == "FAILURE":
        print("failed at the homogeneous stage")
        return FAILURE_OUT
    
    res1_end = res1[1].y[:,-1]

    task_p = (MODELS.PAR3ADD, {**params_par3add, **x_as_dict, "tL": tL_polarise, "initial_condition": res1_end})
    res2 = model_task_handler.run_tasks([task_p])[0]

    if res2[1] == "FAILURE":
        print("failed at the establishment stage")
        return FAILURE_OUT
    
    res2_end = res2[1].y[:,-1]

    res1_a = res1_end[Nx:2*Nx] + res1_end[2*Nx:3*Nx]
    res2_a = res2_end[Nx:2*Nx] + res2_end[2*Nx:3*Nx]

    residue = linalg.vector_norm(args["g_homo"] - np.concatenate((res1_a,res1_end[3*Nx:]))) \
        * linalg.vector_norm(args["g_polarised"] - np.concatenate((res2_a,res2_end[3*Nx:])))
    # residue = linalg.vector_norm(args["g_polarised"] - np.concatenate((res2_a,res2_end[3*Nx:])))
    print(residue)
    return residue


if __name__ == '__main__':
    main()
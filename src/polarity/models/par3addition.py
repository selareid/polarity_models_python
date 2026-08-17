# Expansion on the existing model by Goehring et al. 2011 in order to better represent the endometrial epithelia
import time
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt, animation
from scipy import integrate


def default_v_func(kvals, x, t):
    v_time = 600
    time_factor = 1 / np.maximum(1, t / 10 - v_time / 10)

    center = kvals["xL"] / 4
    sd = np.minimum(center / 4, (kvals["xL"] - center) / 4)
    peak = 0.1

    return time_factor * peak * np.exp(-(x - center) ** 2 / (2 * sd ** 2))


Ybar = lambda kvals, Y: 2 * integrate.simpson(Y, x = kvals["X"]) / kvals["L"]  # all of J,A,P-bar
def J_cyto(kvals, J, M): return kvals["rho_J"] - kvals["psi"] * Ybar(kvals, J) \
        - kvals["psi"] * Ybar(kvals, M)
def A_cyto(kvals, A, M): return kvals["rho_A"] - kvals["psi"] * Ybar(kvals, A) \
        - kvals["psi"] * Ybar(kvals, M)
def P_cyto(kvals, P): return kvals["rho_P"] - kvals["psi"] * Ybar(kvals, P)


DEFAULT_PARAMETERS = {
    "label": "par3addition",
    "points_per_second": 0.01,

    # General Setup Variables
    "Nx": 100,  # number of length steps
    "L": 134.6,  # length of region
    "x0": 0,
    "xL": 67.3,  # L / 2
    "t0": 0,
    "tL": 9000,

    "v_func": default_v_func,
    
    # Model parameters
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


def disc_diffusion_term(kvals: dict, Y, x_i):
    # This function accounts for boundary reflection
    if x_i == 0:  # left boundary
        return (Y[1] - 2 * Y[0] + Y[1]) / kvals["deltax"] ** 2  # reflect Y[-1] to Y[1]
    elif x_i == kvals["Nx"] - 1:  # right boundary
        return (Y[kvals["Nx"] - 2] - 2 * Y[kvals["Nx"] - 1] + Y[kvals["Nx"] - 2]) / kvals[
            "deltax"] ** 2  # reflect Y[Nx] over Nx-1 to Y[Nx-2]
    else:  # internal point
        return (Y[x_i + 1] - 2 * Y[x_i] + Y[x_i - 1]) / kvals["deltax"] ** 2


# where func is a function of type x_i -> float
def disc_spatial_derivative(kvals: dict, func: Callable[[int], float], x_i):
    return (func(x_i + 1) - func(x_i)) / kvals["deltax"]


R_J = lambda kvals, J, M, A, P, t, x_i, A_cyto_r, J_cyto_r: -kvals["k1"]*A_cyto_r*J[x_i] + kvals["k2"]*M[x_i] \
                                                    + kvals["konJ"]*J_cyto_r - kvals["koffJ"]*J[x_i] \
                                                    - kvals["kJP"]*P[x_i]**kvals["alpha"]*J[x_i]
R_M = lambda kvals, J, M, A, P, t, x_i, A_cyto_r: kvals["k1"]*A_cyto_r*J[x_i] - kvals["k2"]*M[x_i] \
                                                    - kvals["koffM"]*M[x_i] \
                                                    - kvals["kMP"]*P[x_i]*M[x_i]  # added antagonism
R_A = lambda kvals, J, M, A, P, t, x_i, A_cyto_r: kvals["k2"]*M[x_i] + kvals["konA"]*A_cyto_r - kvals["koffA"]*A[x_i] \
                                                    - kvals["kAP"]*P[x_i]*A[x_i]  # added antagonism
R_P = lambda kvals, J, M, A, P, t, x_i, P_cyto_r: kvals["konP"]*P_cyto_r - kvals["koffP"]*P[x_i] \
                                                    - kvals["kPA"]*(A[x_i]+M[x_i])**kvals["beta"]*P[x_i]


def odefunc(t, U, kvals):
    Nx = kvals["Nx"]

    assert len(U) == 4 * Nx

    # Failure so odefunc doesn't run forever trying to fix numerical issues
    if min(U) < -100 or max(U) > 100:
        print(f"FAILURE with par3addition labelled {kvals['label']} at simulation time {t:.4f}")
        # plot_failure(U, t, kvals)
        raise AssertionError

    J = U[:Nx]
    M = U[Nx:2*Nx]
    A = U[2*Nx:3*Nx]
    P = U[3*Nx:]

    dudt_J = [0]*Nx
    dudt_M = [0]*Nx
    dudt_A = [0]*Nx
    dudt_P = [0]*Nx

    # r is for "resolved"
    J_cyto_r = J_cyto(kvals, J, M)
    A_cyto_r = A_cyto(kvals, A, M)
    P_cyto_r = P_cyto(kvals, P)

    # insides
    # diffusion function handles left boundary
    for x_i in np.arange(0, Nx-1):
        dudt_J[x_i] = kvals["D_J"]*disc_diffusion_term(kvals, J, x_i) \
                        -kvals["sigmaJ"]*disc_spatial_derivative(kvals, lambda x_ii: kvals["v_func"](kvals, kvals["X"][x_ii], t)*J[x_ii], x_i) \
                        + R_J(kvals, J, M, A, P, t, x_i, A_cyto_r, J_cyto_r)
        dudt_M[x_i] = kvals["D_M"]*disc_diffusion_term(kvals, M, x_i) \
                        -kvals["sigmaM"]*disc_spatial_derivative(kvals, lambda x_ii: kvals["v_func"](kvals, kvals["X"][x_ii], t)*M[x_ii], x_i) \
                        + R_M(kvals, J, M, A, P, t, x_i, A_cyto_r)
        dudt_A[x_i] = kvals["D_A"]*disc_diffusion_term(kvals, A, x_i) \
                        + R_A(kvals, J, M, A, P, t, x_i, A_cyto_r)
        dudt_P[x_i] = kvals["D_P"]*disc_diffusion_term(kvals, P, x_i) \
                        -kvals["sigmaP"]*disc_spatial_derivative(kvals, lambda x_ii: kvals["v_func"](kvals, kvals["X"][x_ii], t)*P[x_ii], x_i) \
                        + R_P(kvals, J, M, A, P, t, x_i, P_cyto_r)

    # manually handle right boundary ( x_i = Nx-1 ) since v(x,t) is odd
    # reflect Nx over Nx-1 to Nx-2; for v_func, also negate on the reflection as v(x)=-v(-x)
    x_i = Nx-1
    dudt_J[x_i] = kvals["D_J"]*disc_diffusion_term(kvals, J, x_i) \
                    - (-kvals["v_func"](kvals, kvals["X"][Nx-2], t)*J[Nx-2] - kvals["v_func"](kvals, kvals["X"][Nx-1], t)*J[Nx-1]) / kvals["deltax"] \
                    + R_J(kvals, J, M, A, P, t, x_i, A_cyto_r, J_cyto_r)
    dudt_M[x_i] = kvals["D_M"]*disc_diffusion_term(kvals, M, x_i) \
                    - (-kvals["v_func"](kvals, kvals["X"][Nx-2], t)*M[Nx-2] - kvals["v_func"](kvals, kvals["X"][Nx-1], t)*M[Nx-1]) / kvals["deltax"] \
                    + R_M(kvals, J, M, A, P, t, x_i, A_cyto_r)
    dudt_A[x_i] = kvals["D_A"]*disc_diffusion_term(kvals, A, x_i) \
                    + R_A(kvals, J, M, A, P, t, x_i, A_cyto_r)
    dudt_P[x_i] = kvals["D_P"]*disc_diffusion_term(kvals, P, x_i) \
                    - (-kvals["v_func"](kvals, kvals["X"][Nx-2], t)*P[Nx-2] - kvals["v_func"](kvals, kvals["X"][Nx-1], t)*P[Nx-1]) / kvals["deltax"] \
                    + R_P(kvals, J, M, A, P, t, x_i, P_cyto_r)

    return dudt_J + dudt_M + dudt_A + dudt_P


def run_model(args=None):
    if args is None:
        args = {}
    params = {**DEFAULT_PARAMETERS, **args}

    # calculate other widely used values
    X = np.linspace(params["x0"], params["xL"], params["Nx"])
    deltax = np.abs(X[1] - X[0])

    # key values
    kvals: dict = {**params, "X": X, "deltax": deltax}

    # default time points for solver output
    kvals["t_eval"] = kvals["t_eval"] if "t_eval" in kvals else np.linspace(kvals["t0"], kvals["tL"], int(kvals["points_per_second"] * np.abs(kvals["tL"] - kvals["t0"])))

    # default initial condition (just all 0) if none passed
    kvals["initial_condition"] = kvals["initial_condition"] if "initial_condition" in kvals else [0]*(kvals["Nx"]*4)

    sol = integrate.solve_ivp(odefunc, [kvals["t0"], kvals["tL"]], kvals["initial_condition"], method="BDF",
                              t_eval=kvals["t_eval"], args=(kvals,))

    return sol, kvals

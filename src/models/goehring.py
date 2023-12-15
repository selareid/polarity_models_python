# Based on Goehring et al. 2011
import time
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt, animation
from scipy import integrate

def default_v_func(kvals, x,t):
    if t > 1000:
        return 0*x
    else:
        return -x*(x-190)*(x-35) / (700000*(np.maximum(1,np.abs((t-120)/80)))**2)


Ybar = lambda kvals, Y: 2 * integrate.simpson(Y, kvals["X"]) / kvals["L"]  # handles both A-bar and P-bar
def default_A_cyto(kvals, A): return kvals["rho_A"] - kvals["psi"] * Ybar(kvals, A)
def default_P_cyto(kvals, P): return kvals["rho_P"] - kvals["psi"] * Ybar(kvals, P)

DEFAULT_PARAMETERS = {
    "label": "goehring",
    "points_per_second": 2,

    # General Setup Variables
    "Nx": 100,  # number of length steps
    "L": 134.6,  # length of region
    "x0": 0,
    "xL": 67.3,  # L / 2
    "t0": 0,
    "tL": 5000,

    # Model parameters and functions
    "psi": 0.174,
    "D_A": 0.28,
    "D_P": 0.15,
    "k_onA": 8.58 * 10 ** (-3),
    "k_onP": 4.74 * 10 ** (-2),  # (TODO) 10^-3 or 10^-2 ?? (is 2 in the paper)
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
    "A_cyto": default_A_cyto,
    "P_cyto": default_P_cyto,
    "v_func": default_v_func,
}


def discrete_diffusion_term(kvals: dict, Y, x_i):
    # This function accounts for boundary reflection
    if x_i == 0:  # left boundary
        return (Y[1] - 2 * Y[0] + Y[1]) / kvals["deltax"] ** 2  # reflect Y[-1] to Y[1]
    elif x_i == kvals["Nx"] - 1:  # right boundary
        return (Y[kvals["Nx"] - 2] - 2 * Y[kvals["Nx"] - 1] + Y[kvals["Nx"] - 2]) / kvals[
            "deltax"] ** 2  # reflect Y[Nx] over Nx-1 to Y[Nx-2]
    else:  # internal point
        return (Y[x_i + 1] - 2 * Y[x_i] + Y[x_i - 1]) / kvals["deltax"] ** 2


# where func is a function of type x_i -> float
def discrete_spatial_derivative(kvals: dict, func: Callable[[int], float], x_i):
    return (func(x_i + 1) - func(x_i)) / kvals["deltax"]

R_A = lambda kvals, A, P, A_cyto_r, t, x_i: kvals["k_onA"] * A_cyto_r \
    - kvals["k_offA"] * A[x_i] - kvals["k_AP"] * (P[x_i] ** kvals["alpha"]) * A[x_i]
R_P = lambda kvals, A, P, P_cyto_r, t, x_i: kvals["k_onP"] * P_cyto_r \
    - kvals["k_offP"] * P[x_i] - kvals["k_PA"] * (A[x_i] ** kvals["beta"]) * P[x_i]


def odefunc(t, U, kvals):
    assert len(U) == 2 * kvals["Nx"]

    A = U[:kvals["Nx"]]
    P = U[kvals["Nx"]:]

    dudt_A = np.zeros(kvals["Nx"])
    dudt_P = np.zeros(kvals["Nx"])

    # r is for "resolved"
    A_cyto_r = kvals["A_cyto"](kvals, A)
    P_cyto_r = kvals["P_cyto"](kvals, P)

    # manually handle right boundary ( x_i = Nx-1 ) since v(x,t)
    # reflect Nx over Nx-1 to Nx-2; for v_func, also negate on the reflection as v(x)=-v(-x)
    dudt_A[kvals["Nx"]-1] = kvals["D_A"] * discrete_diffusion_term(kvals, A, kvals["Nx"]-1) \
        - (kvals["v_func"](kvals, kvals["X"][kvals["Nx"]-2], t) * A[kvals["Nx"]-2] - kvals["v_func"](kvals, kvals["X"][kvals["Nx"]-1], t) * A[kvals["Nx"]-1]) / kvals["deltax"] \
        + R_A(kvals, A, P, A_cyto_r, t, kvals["Nx"]-1)
    dudt_P[kvals["Nx"]-1] = kvals["D_P"] * discrete_diffusion_term(kvals, P, kvals["Nx"]-1) \
        - (kvals["v_func"](kvals, kvals["X"][kvals["Nx"]-2], t) * P[kvals["Nx"]-2] - kvals["v_func"](kvals, kvals["X"][kvals["Nx"]-1], t) * P[kvals["Nx"]-1]) / kvals["deltax"] \
        + R_P(kvals, A, P, P_cyto_r, t, kvals["Nx"]-1)

    # insides
    # diffusion function handles left boundary
    for x_i in np.arange(0, kvals["Nx"] - 1):
        dudt_A[x_i] = kvals["D_A"] * discrete_diffusion_term(kvals, A, x_i) \
              - discrete_spatial_derivative(kvals, lambda x_ii: kvals["v_func"](kvals, kvals["X"][x_ii], t) * A[x_ii], x_i) \
              + R_A(kvals, A, P, A_cyto_r, t, x_i)
        dudt_P[x_i] = kvals["D_P"] * discrete_diffusion_term(kvals, P, x_i) \
              - discrete_spatial_derivative(kvals, lambda x_ii: kvals["v_func"](kvals, kvals["X"][x_ii], t) * P[x_ii], x_i) \
              + R_P(kvals, A, P, P_cyto_r, t, x_i)

    return np.ravel([dudt_A, dudt_P])


def run_model(args: dict = {}):
    params = {**DEFAULT_PARAMETERS, **args}

    # calculate other widely used values
    X = np.linspace(params["x0"], params["xL"], params["Nx"])
    deltax = np.abs(X[1] - X[0])

    # key values
    kvals: dict = {**params, "X": X, "deltax": deltax}

    # default time points for solver output
    kvals["t_eval"] = kvals["t_eval"] if "t_eval" in kvals else np.linspace(kvals["t0"], kvals["tL"], int(kvals["points_per_second"] * np.abs(kvals["tL"] - kvals["t0"])))

    # default initial condition if none passed
    kvals["initial_condition"] = kvals["initial_condition"] if "initial_condition" in kvals else np.ravel([[0.5, 0.0] for x_i in np.arange(0, kvals["Nx"])], order='F')

    sol = integrate.solve_ivp(odefunc, [kvals["t0"], kvals["tL"]], kvals["initial_condition"], method="BDF",
                              t_eval=kvals["t_eval"], args=(kvals,))

    return sol, kvals

# Plotting
def animate_plot(sol, kvals: dict, save_file = False, file_code: str = None):
    if file_code is None:
        file_code = f'{time.time_ns()}'[5:]

    fig, ax = plt.subplots()
    line1, = ax.plot(kvals["X"], sol.y[:kvals["Nx"], 0], label="anterior")
    line2, = ax.plot(kvals["X"], sol.y[kvals["Nx"]:, 0], label="posterior")
    time_label = ax.text(0.1, 1.05, f"t={sol.t[0]}", transform=ax.transAxes, ha="center")
    linev, = ax.plot(kvals["X"], [kvals["v_func"](kvals, x, 0) for x in kvals["X"]], label="v", linestyle="--")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y)-0.05,np.max(sol.y)+0.05], xlabel="x", ylabel="A/P")
    ax.legend()

    def animate(t_i):
        linev.set_ydata([kvals["v_func"](kvals, x, 0) for x in kvals["X"]])
        line1.set_ydata(sol.y[:kvals["Nx"], t_i])
        line2.set_ydata(sol.y[kvals["Nx"]:, t_i])
        time_label.set_text(f"t={sol.t[t_i]:.2f}")
        return (line1, line2, linev, time_label)

    ani = animation.FuncAnimation(fig, animate, interval=20 * np.abs(kvals["tL"] - kvals["t0"]) / len(sol.t), blit=True, frames=len(sol.t))

    if save_file:
        file_name = f"{file_code}_spatialPar.mp4"
        print(f"Saving animation to {file_name}")
        ani.save(file_name)

    plt.show(block=False)

def plot_final_timestep(sol, kvals):
    plt.figure()
    ax = plt.subplot()

    ax.plot(kvals["X"], sol.y[:kvals["Nx"], -1], label="anterior")
    ax.plot(kvals["X"], sol.y[kvals["Nx"]:, -1], label="posterior")
    ax.text(0.1, 1.05, f"t={sol.t[-1]}", transform=ax.transAxes, ha="center")
    ax.plot(kvals["X"], [kvals["v_func"](kvals, x, -1) for x in kvals["X"]], label="v", linestyle="--")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y[:, -1])-0.05, np.max(sol.y[:, -1])+0.05], xlabel="x", ylabel="A/P")
    ax.legend()

    plt.show(block=False)


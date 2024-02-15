# Expansion on the existing model by Goehring et al. 2011 in order to better represent the endometrial epithelia
import time
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt, animation
from scipy import integrate

from src.models.metric_functions import polarity_measure, orientation_marker, polarity_orientation, polarity_get_all


def default_v_func(kvals, x, t):
    v_time = 600
    time_factor = 1 / np.maximum(1, t / 10 - v_time / 10)

    center = kvals["xL"] / 4
    sd = np.minimum(center / 4, (kvals["xL"] - center) / 4)
    peak = 0.1

    return time_factor * peak * np.exp(-(x - center) ** 2 / (2 * sd ** 2))


Ybar = lambda kvals, Y: 2 * integrate.simpson(Y, kvals["X"]) / kvals["L"]  # all of J,A,P-bar
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
    "tL": 5000,

    # Model parameters and functions #
    "psi": 0.174,  # surface to volume conversion factor

    # diffusion coefficients
    "D_J": 0.28,
    "D_M": 0.28,
    "D_A": 0.28,
    "D_P": 0.15,

    # velocity coefficients
    "sigmaJ": 1,
    "sigmaM": 1,
    "sigmaP": 1,

    # aPar complex change coefficients
    "k1": 8.58 * 10**(-3),
    "k2": 6.5 * 10**(-3),

    # antagonism
    "kJP": 0.190,
    "kPA": 2.0,

    # antagonism exponents
    "alpha": 1,
    "beta": 2,

    # total amount in system
    "rho_A": 1.56,
    "rho_J": 1.56,
    "rho_P": 1.0,

    "konJ": 8.58 * 10**(-3),
    "konA": 0,
    "konP": 4.74 * 10**(-2),

    "koffJ": 7.3 * 10**(-3),
    "koffA": 5.4 * 10**(-3),
    "koffP": 7.3 * 10**(-3),

    "v_func": default_v_func,

    # these two added later
    "kMP": 0,
    "kAP": 0,
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


# Plotting
def animate_plot(sol, kvals: dict, save_file=False, file_code: str = None, rescale=False):
    if file_code is None:
        file_code = f'{time.time_ns()}'[5:]

    # rescale so maximal protein quantity is 1
    scalar = 1 if not rescale else np.max(sol.y)
    v_rescale_for_visibility = np.max(sol.y)/scalar * 10  # rescale so 0.1 is equal to max protein quantity in the plotting of v

    Nx = kvals["Nx"]
    # J = U[:Nx]
    # M = U[Nx:2 * Nx]
    # A = U[2 * Nx:3 * Nx]
    # P = U[3 * Nx:]
    fig, ax = plt.subplots()
    line1, = ax.plot(kvals["X"], sol.y[:Nx, 0]/scalar, label="par3", color="green")
    line2, = ax.plot(kvals["X"], sol.y[Nx:2*Nx, 0]/scalar, label="par3-PKC", color="purple")
    line3, = ax.plot(kvals["X"], sol.y[2*Nx:3*Nx, 0]/scalar, label="cdc42-PKC", color="blue")
    line4, = ax.plot(kvals["X"], sol.y[3*Nx:, 0]/scalar, label="posterior", color="orange")
    p_m, _, _ = polarity_get_all(kvals["X"], sol.y[2*Nx:3*Nx, 0], sol.y[3*Nx:, 0], Nx)
    time_label = ax.text(0.1, 1.05, f"t={sol.t[0]} p={p_m:.4f}", transform=ax.transAxes, ha="center")
    linev, = ax.plot(kvals["X"], [v_rescale_for_visibility*kvals["v_func"](kvals, x, 0) for x in kvals["X"]], label="v", linestyle="--", color="black")

    ax.text(0.7, 1.05, kvals["label"] + ";Nx:" + str(Nx), transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y)/scalar-0.05,np.max(sol.y)/scalar+0.05], xlabel="x", ylabel="par3,A/P")
    ax.legend()

    def animate(t_i):
        linev.set_ydata([v_rescale_for_visibility*kvals["v_func"](kvals, x, sol.t[t_i]) for x in kvals["X"]])
        line1.set_ydata(sol.y[:Nx, t_i]/scalar)
        line2.set_ydata(sol.y[Nx:2*Nx, t_i]/scalar)
        line3.set_ydata(sol.y[2*Nx:3*Nx, t_i]/scalar)
        line4.set_ydata(sol.y[3*Nx:, t_i]/scalar)
        p_m, _, _ = polarity_get_all(kvals["X"], sol.y[2*Nx:3*Nx, t_i], sol.y[3*Nx:, t_i], Nx)
        time_label.set_text(f"t={sol.t[t_i]:.2f} p={p_m:.4f}")
        return (line1, line2, line3, linev, time_label)

    ani = animation.FuncAnimation(fig, animate, interval=10000/len(sol.t), blit=True, frames=len(sol.t))

    if save_file:
        file_name = f"{file_code}_spatialPar.mp4"
        print(f"Saving animation to {file_name}")
        ani.save(file_name)

    plt.show(block=False)


def plot_final_timestep(sol, kvals, rescale=False):
    plt.figure()
    ax = plt.subplot()

    Nx = kvals["Nx"]
    scalar = 1 if not rescale else np.max(sol.y)

    ax.plot(kvals["X"], sol.y[:Nx, -1] / scalar, label="par3", color="green")
    ax.plot(kvals["X"], sol.y[Nx:2 * Nx, -1] / scalar, label="par3-PKC", color="purple")
    ax.plot(kvals["X"], sol.y[2 * Nx:3 * Nx, -1] / scalar, label="cdc42-PKC", color="blue")
    ax.plot(kvals["X"], sol.y[3 * Nx:, -1] / scalar, label="posterior", color="orange")

    p_m, _, _ = polarity_get_all(kvals["X"], sol.y[2*Nx:3*Nx, -1], sol.y[3*Nx:, -1], Nx)
    ax.text(0.1, 1.05, f"t={sol.t[-1]},p={p_m:.4f}", transform=ax.transAxes, ha="center")  # time value
    ax.plot(kvals["X"], [kvals["v_func"](kvals, x, sol.t[-1]) for x in kvals["X"]], label="v", linestyle="--", color="black")  # v_func

    ax.text(0.7, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    # ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y[:, -1])/scalar-0.05, np.max(sol.y[:, -1])/scalar+0.05], xlabel="x", ylabel="A/P")
    ax.legend()

    plt.show(block=False)


# polarity based on the cdc42 quantity for Anterior
# untested code
def plot_variation_sets(variation_sets, label=DEFAULT_PARAMETERS["label"], x_axis_labels: list[str] | None = None, show_orientation=True, xlim=None):
    plt.figure()
    ax = plt.subplot()

    # add then remove plot with xticks so that they get ordered correctly in the figure
    sentinel, = ax.plot(x_axis_labels, [0.5]*len(x_axis_labels))
    sentinel.remove()

    for i in np.arange(0, len(variation_sets)):
        variation = variation_sets[i]
        sol_list = variation[0]
        kvals_list = variation[1]

        polarity_m_list = []
        xticks = []
        if len(variation_sets) > 7:
            color = (np.minimum(1, 0.3 + (i % 6)/7), 0.75 - 0.50*i/len(variation_sets),0.5 + 0.50*i/len(variation_sets))
        else:
            color = (np.minimum(1, 0.3 + (i % 3)/4), 0.75 - 0.50*i/len(variation_sets),0.5 + 0.50*i/len(variation_sets))

        for j in np.arange(0, len(sol_list)):
            sol = sol_list[j]
            kvals = kvals_list[j]

            if not sol == "FAILURE":
                p_measure, _p_orientation, p_marker = polarity_get_all(kvals["X"], sol.y[2*kvals["Nx"]:3*kvals["Nx"], -1], sol.y[3*kvals["Nx"]:, -1], kvals["Nx"])

                xtick = x_axis_labels[j] if x_axis_labels is not None else j
                marker = 'o' if not show_orientation else p_marker

                # jitter the near-0 values so they are visible
                if p_measure<0.02:
                    p_measure += 0.02*i/len(variation_sets)-0.01

                polarity_m_list.append(p_measure)
                xticks.append(xtick)
                ax.scatter(xtick, p_measure, color=color, marker=marker, s=100)

        ax.plot(xticks, polarity_m_list, "--", label=kvals_list[1]["key_varied"], color=color)

    ax.legend()
    ax.set(xlabel="percentage of baseline value", ylabel="polarity", ylim=[-0.1,1.1], xlim=xlim)
    ax.title.set_text(label)
    plt.show(block=False)


# Based on Goehring et al. 2011
import time
from typing import Callable
import numpy as np
from matplotlib import pyplot as plt, animation
from scipy import integrate
from .metric_functions import polarity_measure, polarity_orientation, orientation_marker


def default_v_func(kvals, x, t):
    v_time = 600
    time_factor = 1 / np.maximum(1, t / 10 - v_time / 10)

    center = kvals["xL"] / 4
    sd = np.minimum(center / 4, (kvals["xL"] - center) / 4)
    peak = 0.1

    return time_factor * peak * np.exp(-(x - center) ** 2 / (2 * sd ** 2))


Ybar = lambda kvals, Y: 2 * integrate.simpson(Y, x = kvals["X"]) / kvals["L"]  # handles both A-bar and P-bar
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
    "tL": 9000,

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
    "A_cyto": default_A_cyto,
    "P_cyto": default_P_cyto,
    "v_func": default_v_func,
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

R_A = lambda kvals, A, P, A_cyto_r, t, x_i: kvals["k_onA"] * A_cyto_r \
    - kvals["k_offA"] * A[x_i] - kvals["k_AP"] * (P[x_i] ** kvals["alpha"]) * A[x_i]
R_P = lambda kvals, A, P, P_cyto_r, t, x_i: kvals["k_onP"] * P_cyto_r \
    - kvals["k_offP"] * P[x_i] - kvals["k_PA"] * (A[x_i] ** kvals["beta"]) * P[x_i]


def odefunc(t, U, kvals):
    assert len(U) == 2 * kvals["Nx"]

    # Failure so odefunc doesn't run forever trying to fix numerical issues
    if min(U) < -100 or max(U) > 100:
        print(f"FAILURE with goehring labelled {kvals['label']} at simulation time {t:.4f}")
        # plot_failure(U, t, kvals)
        raise AssertionError

    A = U[:kvals["Nx"]]
    P = U[kvals["Nx"]:]

    dudt_A = np.zeros(kvals["Nx"])
    dudt_P = np.zeros(kvals["Nx"])

    # r is for "resolved"
    A_cyto_r = kvals["A_cyto"](kvals, A)
    P_cyto_r = kvals["P_cyto"](kvals, P)

    # manually handle right boundary ( x_i = Nx-1 ) since v(x,t) is odd
    # reflect Nx over Nx-1 to Nx-2; for v_func, also negate on the reflection as v(x)=-v(-x)
    dudt_A[kvals["Nx"]-1] = kvals["D_A"] * disc_diffusion_term(kvals, A, kvals["Nx"]-1) \
        - (-kvals["v_func"](kvals, kvals["X"][kvals["Nx"]-2], t) * A[kvals["Nx"]-2] - kvals["v_func"](kvals, kvals["X"][kvals["Nx"]-1], t) * A[kvals["Nx"]-1]) / kvals["deltax"] \
        + R_A(kvals, A, P, A_cyto_r, t, kvals["Nx"]-1)
    dudt_P[kvals["Nx"]-1] = kvals["D_P"] * disc_diffusion_term(kvals, P, kvals["Nx"]-1) \
        - (-kvals["v_func"](kvals, kvals["X"][kvals["Nx"]-2], t) * P[kvals["Nx"]-2] - kvals["v_func"](kvals, kvals["X"][kvals["Nx"]-1], t) * P[kvals["Nx"]-1]) / kvals["deltax"] \
        + R_P(kvals, A, P, P_cyto_r, t, kvals["Nx"]-1)

    # insides
    # diffusion function handles left boundary
    for x_i in np.arange(0, kvals["Nx"] - 1):
        dudt_A[x_i] = kvals["D_A"] * disc_diffusion_term(kvals, A, x_i) \
              - disc_spatial_derivative(kvals, lambda x_ii: kvals["v_func"](kvals, kvals["X"][x_ii], t) * A[x_ii], x_i) \
              + R_A(kvals, A, P, A_cyto_r, t, x_i)
        dudt_P[x_i] = kvals["D_P"] * disc_diffusion_term(kvals, P, x_i) \
              - disc_spatial_derivative(kvals, lambda x_ii: kvals["v_func"](kvals, kvals["X"][x_ii], t) * P[x_ii], x_i) \
              + R_P(kvals, A, P, P_cyto_r, t, x_i)

    return np.ravel([dudt_A, dudt_P])


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

    # default initial condition if none passed
    kvals["initial_condition"] = kvals["initial_condition"] if "initial_condition" in kvals else np.ravel([[1.51, 0.0] for x_i in np.arange(0, kvals["Nx"])], order='F')

    sol = integrate.solve_ivp(odefunc, [kvals["t0"], kvals["tL"]], kvals["initial_condition"], method="BDF",
                              t_eval=kvals["t_eval"], args=(kvals,))

    return sol, kvals


# Plotting
def animate_plot(sol, kvals: dict, save_file=False, file_code: str = None, rescale=False):
    if file_code is None:
        file_code = f'{time.time_ns()}'[5:]

    # rescale so maximal protein quantity is 1
    scalar = 1 if not rescale else np.max(sol.y)
    v_rescale_for_visibility = np.max(sol.y)/scalar * 10 # rescale so 0.1 is equal to max protein quantity in the plotting of v


    fig, ax = plt.subplots()
    line1, = ax.plot(kvals["X"], sol.y[:kvals["Nx"], 0]/scalar, label="anterior", color="blue")
    line2, = ax.plot(kvals["X"], sol.y[kvals["Nx"]:, 0]/scalar, label="posterior", color="orange")
    p_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], 0], sol.y[kvals["Nx"]:, 0], kvals["Nx"])
    time_label = ax.text(0.1, 1.05, f"t={sol.t[0]} p={p_m:.4f}", transform=ax.transAxes, ha="center")
    linev, = ax.plot(kvals["X"], [v_rescale_for_visibility*kvals["v_func"](kvals, x, 0) for x in kvals["X"]], label="v", linestyle="--", color="black")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y)/scalar-0.05,np.max(sol.y)/scalar+0.05], xlabel="x", ylabel="A/P")
    ax.legend()

    def animate(t_i):
        linev.set_ydata([v_rescale_for_visibility*kvals["v_func"](kvals, x, sol.t[t_i]) for x in kvals["X"]])
        line1.set_ydata(sol.y[:kvals["Nx"], t_i]/scalar)
        line2.set_ydata(sol.y[kvals["Nx"]:, t_i]/scalar)
        p_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], t_i], sol.y[kvals["Nx"]:, t_i], kvals["Nx"])
        time_label.set_text(f"t={sol.t[t_i]:.2f} p={p_m:.4f}")
        return (line1, line2, linev, time_label)

    ani = animation.FuncAnimation(fig, animate, interval=5000/len(sol.t), blit=True, frames=len(sol.t))

    if save_file:
        file_name = f"{file_code}_spatialPar.mp4"
        print(f"Saving animation to {file_name}")
        ani.save(file_name)

    plt.show(block=False)


def plot_final_timestep(sol, kvals, rescale=False):
    plt.figure()
    ax = plt.subplot()

    scalar = 1 if not rescale else np.max(sol.y)

    ax.plot(kvals["X"], sol.y[:kvals["Nx"], -1]/scalar, label="anterior", color="blue") # A
    ax.plot(kvals["X"], sol.y[kvals["Nx"]:, -1]/scalar, label="posterior", color="orange") # P

    p_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], -1], sol.y[kvals["Nx"]:, -1], kvals["Nx"])
    ax.text(0.1, 1.05, f"t={sol.t[-1]},p={p_m:.4f}", transform=ax.transAxes, ha="center") # time value
    ax.plot(kvals["X"], [kvals["v_func"](kvals, x, sol.t[-1]) for x in kvals["X"]], label="v", linestyle="--", color="black") # v_func

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y[:, -1])/scalar-0.05, np.max(sol.y[:, -1])/scalar+0.05], xlabel="x", ylabel="A/P")
    ax.legend()

    plt.show(block=False)


# this one is different from others cause I wrote it for my poster plots and I didn't generify it
def plot_timestep(sol, kvals, t_i=0, rescale=False, upperYTick=None, show_legend=True, left_ticks=True, ax=None):
    if ax is None:
        plt.figure()
        ax = plt.subplot()

    COLOR_APAR = (0, 150/255, 150/255)
    COLOR_PPAR = (230/255, 0, 0)
    LINE_WIDTH = 5

    scalar = 1 if not rescale else np.max(sol.y[:, t_i])

    ax.plot(kvals["X"], sol.y[:kvals["Nx"], t_i]/scalar, label="Anterior", color=COLOR_APAR, linewidth=LINE_WIDTH) # A
    ax.plot(kvals["X"], sol.y[kvals["Nx"]:, t_i]/scalar, label="Posterior", color=COLOR_PPAR, linewidth=LINE_WIDTH) # P

    # p_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], t_i], sol.y[kvals["Nx"]:, t_i], kvals["Nx"])
    # ax.text(0.1, 1.05, f"t={sol.t[t_i]},p={p_m:.4f}", transform=ax.transAxes, ha="center") # time value

    ax.text(0.1, 1.05, f"t={sol.t[t_i]:.0f}", transform=ax.transAxes, ha="center") # time value


    v_scalar = upperYTick/0.1 if upperYTick is not None else 1/0.1 if rescale else 1
    ax.plot(kvals["X"], [v_scalar*kvals["v_func"](kvals, x, sol.t[t_i]) for x in kvals["X"]], label="v", linestyle="--", color="black", linewidth=LINE_WIDTH) # v_func

    # ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y[:, t_i])/scalar-0.05, np.max(sol.y[:, t_i])/scalar+0.05])#, xlabel="x", ylabel="A/P")
    
    if show_legend: ax.legend()

    plt.xticks([0,kvals["xL"]//2,kvals["xL"]],['0','','L'])

    if upperYTick is not None:
        plt.yticks([0, upperYTick], [0,1] if left_ticks else ['',''])

    plt.show(block=False)


# plot cytoplasmic quantities over time
def plot_cyto(sol, kvals):
    plt.figure()
    ax = plt.subplot()

    ax.plot(sol.t, [kvals["A_cyto"](kvals, sol.y[:kvals["Nx"], t_i]) for t_i in np.arange(0, len(sol.t))], label="A_cyto", color="blue")
    ax.plot(sol.t, [kvals["P_cyto"](kvals, sol.y[kvals["Nx"]:, t_i]) for t_i in np.arange(0, len(sol.t))], label="P_cyto", color="orange")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlabel="time")

    ax.title.set_text("Cytoplasmic Quantities")

    ax.legend()
    plt.show(block=False)

def plot_overall_quantities_over_time(sol, kvals, rescale_by_length=True):
    plt.figure()
    ax = plt.subplot()

    # since this is overall quantity, rescale by space length
    length_scalar = 1 if not rescale_by_length else np.abs(kvals["xL"] - kvals["x0"])

    #TODO - unsure if I should plot with or without the psi multiple
    ax.plot(sol.t, [kvals["A_cyto"](kvals, sol.y[:kvals["Nx"], t_i])/length_scalar for t_i in np.arange(0, len(sol.t))],
            label="A_cyto", color="blue", linestyle="--")
    ax.plot(sol.t, [kvals["P_cyto"](kvals, sol.y[kvals["Nx"]:, t_i])/length_scalar for t_i in np.arange(0, len(sol.t))],
            label="P_cyto", color="orange", linestyle="--")

    ax.plot(sol.t, [Ybar(kvals, sol.y[:kvals["Nx"], t_i])/length_scalar for t_i in np.arange(0, len(sol.t))], label="A_bar", color="blue")
    ax.plot(sol.t, [Ybar(kvals, sol.y[kvals["Nx"]:, t_i])/length_scalar for t_i in np.arange(0, len(sol.t))], label="P_bar", color="orange")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlabel="time")

    ax.title.set_text("Quantities")

    ax.legend()
    plt.show(block=False)


# plot a bunch of different solutions final timestep (just A,P) on single figure
# Assumes that all solutions have the same X,Nx,x0,xL, and time points
def plot_multi_final_timestep(sol_list, kvals_list, label=DEFAULT_PARAMETERS["label"], plot_A=True, plot_P=True):
    kvals = kvals_list[0]

    plt.figure()
    ax = plt.subplot()

    for i in np.arange(0,len(sol_list)):
        sol = sol_list[i]
        kvals_this_sol = kvals_list[i]

        if plot_A:
            ax.plot(kvals["X"], sol.y[:kvals["Nx"], -1], label=f"A_{kvals_this_sol['label']}", color=(0.3 + (i % 3)/4, 0.75 - 0.50*i/len(sol_list),0.5 + 0.50*i/len(sol_list)))
        if plot_P:
            ax.plot(kvals["X"], sol.y[kvals["Nx"]:, -1], label=f"P_{kvals_this_sol['label']}", color=(0.3 + (i % 3)/4, 0.75 - 0.50*i/len(sol_list),0.5 + 0.50*i/len(sol_list)))

    ax.text(0.1, 1.05, f"t={sol_list[0].t[-1]}", transform=ax.transAxes, ha="center") # timestamp
    ax.text(1, 1.05, label, transform=ax.transAxes, ha="center") # label

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min([sol.y[:, -1] for sol in sol_list])-0.05, np.max([sol.y[:, -1] for sol in sol_list])+0.05], xlabel="x", ylabel="A/P")
    ax.title.set_text("Multiple Sims")
    ax.legend()

    plt.show(block=False)

def plot_failure(U, t, kvals):
    plt.figure()
    ax = plt.subplot()

    ax.plot(kvals["X"], U[:kvals["Nx"]], label="anterior", color="blue")
    ax.plot(kvals["X"], U[kvals["Nx"]:], label="posterior", color="orange")
    ax.text(0.1, 1.05, f"t={t}", transform=ax.transAxes, ha="center")
    ax.plot(kvals["X"], [kvals["v_func"](kvals, x, t) for x in kvals["X"]], label="v", linestyle="--", color="black")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(U)-0.05, np.max(U)+0.05], xlabel="x", ylabel="A/P")
    ax.title.set_text("Failure Plot")
    ax.legend()

    plt.show(block=True)


# assume lists are [base, ...others]
def plot_metric_comparisons(sol_list, kvals_list, label=DEFAULT_PARAMETERS["label"]):
    assert len(sol_list) == len(kvals_list)

    # kvals = kvals_list[0]
    # polarity measure metric (final timestep)
    # plt.figure()
    # ax1 = plt.subplot()

    # TODO other metric
    # plt.figure()
    # ax2 = plt.subplot()

    plt.figure()

    for i in np.arange(0, len(sol_list)):
        sol = sol_list[i]
        kvals = kvals_list[i]

        polarity_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], -1], sol.y[kvals["Nx"]:, -1], kvals["Nx"])

        plt.plot(0, polarity_m, marker="o", linestyle="None", label=kvals["label"])

    plt.legend()
    # plt.title.set_text(label)
    plt.show(block=False)


# variation_sets is a list [([sol,sol,sol], [kvals, kvals, kvals]), ... ]
# assumes kvals has key_varied property
# can handle sol with value "FAILURE"
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
                polarity_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], -1], sol.y[kvals["Nx"]:, -1], kvals["Nx"])
                xtick = x_axis_labels[j] if x_axis_labels is not None else j
                marker = 'o' if not show_orientation else orientation_marker(polarity_orientation(kvals["X"], sol.y[:kvals["Nx"], -1], sol.y[kvals["Nx"]:, -1], kvals["Nx"]))

                # jitter the near-0 values so they are visible
                if polarity_m<0.02:
                    polarity_m += 0.02*i/len(variation_sets)-0.01

                polarity_m_list.append(polarity_m)
                xticks.append(xtick)
                ax.scatter(xtick, polarity_m, color=color, marker=marker, s=100)

        ax.plot(xticks, polarity_m_list, "--", label=kvals_list[1]["key_varied"], color=color)

    ax.legend()
    ax.set(xlabel="percentage of baseline value", ylabel="polarity", ylim=[-0.1,1.1], xlim=xlim)
    ax.title.set_text(label)
    ax.tick_params(which="both", labelsize=15)
    ax.tick_params(axis='x', labelrotation=60)
    plt.show(block=False)


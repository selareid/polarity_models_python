# Based on Tostevin, Howard (2008)
import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .metric_functions import polarity_measure, polarity_orientation, orientation_marker

def default_a_func(kvals, lt, x):
    if 0 <= x <= lt:
        return kvals["a_0"] * kvals["L"] / lt
    else:
        return 0


DEFAULT_PARAMETERS = {
    "label": "tostevin",
    "points_per_second": 2,
    
    # General Setup Variables
    "Nx": 150,  # number of length steps
    "L": 50.0,  # length of region - model parameter
    "x0": 0,
    "xL": 50,  # L
    "t0": 0,
    "tL": 3600,

    # Model parameters and functions
    # Taken from pg3 in paper, used for Figs2,3
    "Dm": 0.25,
    "Dc": 5,
    "cA1": 0.01,
    "cA2": 0.07,
    "cA3": 0.01,
    "cA4": 0.11,
    "cP1": 0.08,
    "cP3": 0.04,
    "cP4": 0.13,
    "lambda_0": 42.5,
    "lambda_1": 27,
    "a_0": 1,
    "epsilon": 0.4,

    "a_func": default_a_func
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


def odefunc(t, U, kvals):
    assert len(U) == 4 * kvals["Nx"] + 1

    # Failure so odefunc doesn't run forever trying to fix numerical issues
    if min(U) < -100 or max(U) > 100:
        print(f"FAILURE with tostevin labelled {kvals['label']} at simulation time {t:.4f}")
        # plot_failure(U, t, kvals)
        raise AssertionError

    Am = U[:kvals["Nx"]]
    Ac = U[kvals["Nx"]:2 * kvals["Nx"]]
    Pm = U[2 * kvals["Nx"]:3 * kvals["Nx"]]
    Pc = U[3 * kvals["Nx"]:4 * kvals["Nx"]]
    l = U[-1]

    dudt_Am = np.zeros(kvals["Nx"])
    dudt_Ac = np.zeros(kvals["Nx"])
    dudt_Pm = np.zeros(kvals["Nx"])
    dudt_Pc = np.zeros(kvals["Nx"])

    # calculate dudt_L
    m_t = (integrate.trapezoid([Am[x_i] for x_i in np.arange(0, kvals["Nx"]) if kvals["X"][x_i] < l],
                               dx=kvals["deltax"])) / kvals["L"]
    lambda_t = kvals["lambda_0"] - kvals["lambda_1"] * m_t
    dudt_L = -kvals["epsilon"] * (l - lambda_t) / lambda_t

    # boundary conditions x=0,x=Nx-1 handled here since only relevant to the diffusion term
    for x_i in np.arange(0, kvals["Nx"]):
        dudt_Am[x_i] = kvals["Dm"] * discrete_diffusion_term(kvals, Am, x_i) \
                       + (kvals["cA1"] + kvals["cA2"] * kvals["a_func"](kvals, l, kvals["X"][x_i])) * Ac[x_i] \
                       - kvals["cA3"] * Am[x_i] - kvals["cA4"] * Am[x_i] * Pm[x_i]
        dudt_Ac[x_i] = kvals["Dc"] * discrete_diffusion_term(kvals, Ac, x_i) \
                       - (kvals["cA1"] + kvals["cA2"] * kvals["a_func"](kvals, l, kvals["X"][x_i])) * Ac[x_i] \
                       + kvals["cA3"] * Am[x_i] + kvals["cA4"] * Am[x_i] * Pm[x_i]
        dudt_Pm[x_i] = kvals["Dm"] * discrete_diffusion_term(kvals, Pm, x_i) \
                       + kvals["cP1"] * Pc[x_i] - kvals["cP3"] * Pm[x_i] - kvals["cP4"] * Am[x_i] * Pm[x_i]
        dudt_Pc[x_i] = kvals["Dc"] * discrete_diffusion_term(kvals, Pc, x_i) \
                       - kvals["cP1"] * Pc[x_i] + kvals["cP3"] * Pm[x_i] + kvals["cP4"] * Am[x_i] * Pm[x_i]

    return np.append(np.ravel([dudt_Am, dudt_Ac, dudt_Pm, dudt_Pc]), dudt_L)


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
    kvals["t_eval"] = kvals["t_eval"] if "t_eval" in kvals else np.linspace(kvals["t0"], kvals["tL"], int(kvals["points_per_second"]*np.abs(kvals["tL"]-kvals["t0"])))

    # default initial condition if none passed
    kvals["initial_condition"] = kvals["initial_condition"] if "initial_condition" in kvals else np.append(
        np.ravel([[1, 0, 0, 1] for x_i in np.arange(0, kvals["Nx"])], order='F'), kvals["L"])

    sol = integrate.solve_ivp(odefunc, [kvals["t0"], kvals["tL"]], kvals["initial_condition"], method="BDF",
                              t_eval=kvals["t_eval"], args=(kvals,))

    return sol, kvals


# Plotting Functions
def animate_plot(sol, kvals: dict, save_file=False, file_code: str = None, rescale=False):
    if file_code is None:
        file_code = f'{time.time_ns()}'[5:]

    # rescale so maximal protein quantity is 1
    scalar = 1 if not rescale else np.max(sol.y[:-1, :])

    fig, ax = plt.subplots()
    linePc, = ax.plot(kvals["X"], sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], 0]/scalar, label="Pc", color="orange", linestyle="--")
    lineAc, = ax.plot(kvals["X"], sol.y[kvals["Nx"]:2 * kvals["Nx"], 0]/scalar, label="Ac", color="blue", linestyle="--")
    linePm, = ax.plot(kvals["X"], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], 0]/scalar, label="Pm", color="orange")
    lineAm, = ax.plot(kvals["X"], sol.y[:kvals["Nx"], 0]/scalar, label="Am", color="blue")

    p_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], 0], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], 0], kvals["Nx"])
    time_label = ax.text(0.1, 1.05, f"t={sol.t[0]} p={p_m:.4f}", transform=ax.transAxes, ha="center")

    if rescale:
        ax.text(1, 1.05, kvals["label"] + " (quantities scaled)", transform=ax.transAxes, ha="center")
    else:
        ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y[:-1, :])/scalar - 0.05, np.max(sol.y[:-1, :])/scalar + 0.05], xlabel="x", ylabel="A/P")
    ax.legend()

    def animate(t_i):
        lineAm.set_ydata(sol.y[:kvals["Nx"], t_i] / scalar)  # Am
        lineAc.set_ydata(sol.y[kvals["Nx"]:2 * kvals["Nx"], t_i] / scalar)  # Ac
        linePm.set_ydata(sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], t_i] / scalar)  # Pm
        linePc.set_ydata(sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], t_i] / scalar)  # Pc

        p_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], t_i], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], t_i], kvals["Nx"])
        time_label.set_text(f"t={sol.t[t_i]:.2f} p={p_m:.4f}")
        return lineAm, lineAc, linePm, linePc, time_label

    ani = animation.FuncAnimation(fig, animate, interval=5000/len(sol.t), blit=True, frames=len(sol.t))

    if save_file:
        file_name = f"{file_code}_refParModelOut.mp4"
        print(f"Saving animation to {file_name}")
        ani.save(file_name)

    plt.show(block=False)


def plot_lt(sol, kvals):
    plt.figure()
    ax = plt.subplot()
    ax.plot(sol.t, sol.y[-1, :], label="l(t)")
    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")
    plt.xlabel("t")
    plt.ylabel("length")
    plt.show(block=False)


def plot_final_timestep(sol, kvals, rescale=False):
    plt.figure()
    ax = plt.subplot()

    # scaling takes into account all time
    scalar = 1 if not rescale else np.max(sol.y[:-1, :])

    ax.plot(kvals["X"], sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], -1]/scalar, label="Pc", color="orange", linestyle="--")
    ax.plot(kvals["X"], sol.y[kvals["Nx"]:2 * kvals["Nx"], -1]/scalar, label="Ac", color="blue", linestyle="--")
    ax.plot(kvals["X"], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1]/scalar, label="Pm", color="orange")
    ax.plot(kvals["X"], sol.y[:kvals["Nx"], -1]/scalar, label="Am", color="blue")

    p_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"], -1], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1], kvals["Nx"])
    ax.text(0.1, 1.05, f"t={sol.t[-1]},p={p_m:.4f}", transform=ax.transAxes, ha="center")  # time label

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y[:-1, -1])/scalar - 0.05, np.max(sol.y[:-1, -1])/scalar + 0.05],
           xlabel="x",
           ylabel="A/P")
    ax.legend()

    plt.show(block=False)


def plot_overall_quantities_over_time(sol, kvals, rescale_by_length=True):
    plt.figure()
    ax = plt.subplot()

    # since this is overall quantity, rescale by space length
    length_scalar = 1 if not rescale_by_length else np.abs(kvals["xL"] - kvals["x0"])

    ax.plot(sol.t, [integrate.trapezoid(sol.y[:kvals["Nx"], t_i], dx=kvals["deltax"])/length_scalar for t_i in np.arange(0, len(sol.t))], label="Am", color="blue")
    ax.plot(sol.t, [integrate.trapezoid(sol.y[kvals["Nx"]:2 * kvals["Nx"], t_i], dx=kvals["deltax"])/length_scalar for t_i in np.arange(0, len(sol.t))], label="Ac", color="blue", linestyle="--")
    ax.plot(sol.t, [integrate.trapezoid(sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], t_i], dx=kvals["deltax"])/length_scalar for t_i in np.arange(0, len(sol.t))], label="Pm", color="orange")
    ax.plot(sol.t, [integrate.trapezoid(sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], t_i], dx=kvals["deltax"])/length_scalar for t_i in np.arange(0, len(sol.t))], label="Pc", color="orange", linestyle="--")

    ax.plot(sol.t, sol.y[-1, :]/length_scalar, label="l(t)")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlabel="time")

    ax.title.set_text("Quantities")

    ax.legend()
    plt.show(block=False)


def plot_multi_final_timestep(sol_list, kvals_list, label=DEFAULT_PARAMETERS["label"], plot_Am=True, plot_Ac=True, plot_Pm=True, plot_Pc=True):
    assert plot_Am or plot_Ac or plot_Pm or plot_Pc

    kvals = kvals_list[0]

    plt.figure()
    ax = plt.subplot()

    bounds = (np.inf, -np.inf)

    for i in np.arange(0, len(sol_list)):
        sol = sol_list[i]
        kvals_this_sol = kvals_list[i]

        if plot_Am:
            ax.plot(kvals["X"], sol.y[:kvals["Nx"], -1], label=f"Am_{kvals_this_sol['label']}", color=(0.3 + (i % 3)/4, 0.75 - 0.50*i/len(sol_list),0.5 + 0.50*i/len(sol_list)))
            bounds = (np.minimum(np.min(sol.y[:kvals["Nx"], -1]), bounds[0]), np.maximum(np.max(sol.y[:kvals["Nx"], -1]), bounds[1]))
        if plot_Ac:
            ax.plot(kvals["X"], sol.y[kvals["Nx"]:2 * kvals["Nx"], -1], label=f"Ac_{kvals_this_sol['label']}", color=(0.3 + (i % 3)/4, 0.75 - 0.50*i/len(sol_list),0.5 + 0.50*i/len(sol_list)), linestyle="--")
            bounds = (np.minimum(np.min(sol.y[kvals["Nx"]:2 * kvals["Nx"], -1]), bounds[0]), np.maximum(np.max(sol.y[kvals["Nx"]:2 * kvals["Nx"], -1]), bounds[1]))
        if plot_Pm:
            ax.plot(kvals["X"], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1], label=f"Pm_{kvals_this_sol['label']}", color=(0.3 + (i % 3)/4, 0.75 - 0.50*i/len(sol_list),0.5 + 0.50*i/len(sol_list)))
            bounds = (np.minimum(np.min(sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1]), bounds[0]), np.maximum(np.max(sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1]), bounds[1]))
        if plot_Pc:
            ax.plot(kvals["X"], sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], -1], label=f"Pc_{kvals_this_sol['label']}", color=(0.3 + (i % 3)/4, 0.75 - 0.50*i/len(sol_list),0.5 + 0.50*i/len(sol_list)), linestyle="--")
            bounds = (np.minimum(np.min(sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], -1]), bounds[0]), np.maximum(np.max(sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], -1]), bounds[1]))

    ax.text(0.1, 1.05, f"t={sol_list[0].t[-1]}", transform=ax.transAxes, ha="center") # timestamp
    ax.text(1, 1.05, label, transform=ax.transAxes, ha="center") # label

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[bounds[0]-0.05, bounds[1]+0.05], xlabel="x", ylabel="A/P")
    ax.title.set_text("Multiple Sims")
    ax.legend()

    plt.show(block=False)

#     ax.plot(kvals["X"], sol.y[:kvals["Nx"], -1], label="Am", color="blue")
#     ax.plot(kvals["X"], sol.y[kvals["Nx"]:2 * kvals["Nx"], -1], label="Ac", color="blue", linestyle="--")
#     ax.plot(kvals["X"], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1], label="Pm", color="orange")
#     ax.plot(kvals["X"], sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], -1], label="Pc", color="orange", linestyle="--")


def plot_failure(U, t, kvals):
    plt.figure()
    ax = plt.subplot()

    ax.plot(kvals["X"], U[:kvals["Nx"]], label="Am", color="blue")
    ax.plot(kvals["X"], U[kvals["Nx"]:2 * kvals["Nx"]], label="Ac", color="blue", linestyle="--")
    ax.plot(kvals["X"], U[2 * kvals["Nx"]:3 * kvals["Nx"]], label="Pm", color="orange")
    ax.plot(kvals["X"], U[3 * kvals["Nx"]:4 * kvals["Nx"]], label="Pc", color="orange", linestyle="--")

    ax.text(0.1, 1.05, f"t={t}", transform=ax.transAxes, ha="center")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(U[:-1])-0.05, np.max(U[:-1])+0.05], xlabel="x", ylabel="A/P")
    ax.title.set_text("Failure Plot")
    ax.legend()

    plt.show(block=True)


# variation_sets is a list [([sol,sol,sol], [kvals, kvals, kvals]), ... ]
# assumes kvals has key_varied property
def plot_variation_sets(variation_sets, label=DEFAULT_PARAMETERS["label"], x_axis_labels: list[str] | None = None, show_orientation=True, xlim=None):
    plt.figure()
    ax = plt.subplot()

    # add then remove plot with xticks so that they get ordered correctly in the figure
    sentinel, = ax.plot(x_axis_labels, [0.5] * len(x_axis_labels))
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
                polarity_m = polarity_measure(kvals["X"], sol.y[:kvals["Nx"],-1], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1], kvals["Nx"])
                xtick = x_axis_labels[j] if x_axis_labels is not None else j
                marker = 'o' if not show_orientation else orientation_marker(polarity_orientation(kvals["X"], sol.y[:kvals["Nx"],-1], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1], kvals["Nx"]))

                # jitter the near-0 values so they are visible
                if polarity_m < 0.02:
                    polarity_m += 0.02 * i / len(variation_sets) - 0.01

                polarity_m_list.append(polarity_m)
                xticks.append(xtick)
                ax.scatter(xtick, polarity_m, color=color, marker=marker, s=100)

        ax.plot(xticks, polarity_m_list, "--", label=kvals_list[1]["key_varied"], color=color)

    ax.legend()
    ax.set(xlabel="percentage of baseline value", ylabel="polarity", ylim=[-0.1,1.1], xlim=xlim)
    ax.title.set_text(label)
    plt.show(block=False)
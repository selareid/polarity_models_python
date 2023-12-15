# Based on Tostevin, Howard (2008)
import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    # global previous_second
    # if int(t) > previous_second:
    #     print(f"odefunc; t: {t:.2f}")
    #     previous_second = int(t)

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

    # insides
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


def run_model(args: dict = {}):
    params = {**DEFAULT_PARAMETERS, **args}
    # TODO - test that this overrides defaults as expected

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
def animate_plot(sol, kvals: dict, save_file = False, file_code: str = None):
    if file_code is None:
        file_code = f'{time.time_ns()}'[5:]

    fig, ax = plt.subplots()
    linePc, = ax.plot(kvals["X"], sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], 0], label="Pc", color="orange", linestyle="--")
    lineAc, = ax.plot(kvals["X"], sol.y[kvals["Nx"]:2 * kvals["Nx"], 0], label="Ac", color="blue", linestyle="--")
    linePm, = ax.plot(kvals["X"], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], 0], label="Pm", color="orange")
    lineAm, = ax.plot(kvals["X"], sol.y[:kvals["Nx"], 0], label="Am", color="blue")
    time_label = ax.text(0.1, 1.05, f"t={sol.t[0]}", transform=ax.transAxes, ha="center")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y[:-1, :]) - 0.05, np.max(sol.y[:-1, :]) + 0.05],
           xlabel="x",
           ylabel="A/P")
    ax.legend()

    def animate(t_i):
        lineAm.set_ydata(sol.y[:kvals["Nx"], t_i])
        lineAc.set_ydata(sol.y[kvals["Nx"]:2 * kvals["Nx"], t_i])
        linePm.set_ydata(sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], t_i])
        linePc.set_ydata(sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], t_i])

        time_label.set_text(f"t={sol.t[t_i]:.2f}")
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


def plot_final_timestep(sol, kvals):
    plt.figure()
    ax = plt.subplot()

    ax.plot(kvals["X"], sol.y[3 * kvals["Nx"]:4 * kvals["Nx"], -1], label="Pc", color="orange", linestyle="--")
    ax.plot(kvals["X"], sol.y[kvals["Nx"]:2 * kvals["Nx"], -1], label="Ac", color="blue", linestyle="--")
    ax.plot(kvals["X"], sol.y[2 * kvals["Nx"]:3 * kvals["Nx"], -1], label="Pm", color="orange")
    ax.plot(kvals["X"], sol.y[:kvals["Nx"], -1], label="Am", color="blue")

    ax.text(0.1, 1.05, f"t={sol.t[-1]}", transform=ax.transAxes, ha="center")

    ax.text(1, 1.05, kvals["label"], transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y[:-1, -1]) - 0.05, np.max(sol.y[:-1, -1]) + 0.05],
           xlabel="x",
           ylabel="A/P")
    ax.legend()

    plt.show(block=False)


def plot(sol, kvals: dict):
    file_code = f'{time.time_ns()}'[5:]

    animate_plot(sol, kvals, file_code)
    plot_lt(sol, kvals)

    return file_code

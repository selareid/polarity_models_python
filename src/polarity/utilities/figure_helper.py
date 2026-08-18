# Universal thing for figures
# ordering for par3add elements is [J, M, A, P]
# for goehring the ordering is just [A, P]
from pathlib import Path
import numpy as np
from pandas import DataFrame, concat
from matplotlib import pyplot as plt, animation
from dataclasses import is_dataclass, asdict

from polarity.model_enums import MODELS
# from polarity.utilities.metric_functions import polarity_measure

FIGURES_DIR = Path(__file__).resolve().parents[3] / "figures"
DATA_DIR = Path(__file__).resolve().parents[3] / "results"

# Plot style
font_size = 16
label_font_size = 14
line_width = 3
page_width_fig = 12

# colours
plot_colours = ['#377eb8','#4daf4a','#984ea3','#ff7f00', "#36454f", '#e41a1c']
colours_map = dict(zip(["J", "M", "A", "P", "polarity", "other"], plot_colours))
cmap_polarity = plt.cm.colors.LinearSegmentedColormap.from_list("", [colours_map["polarity"], "white"])
cmap_interface = plt.cm.colors.LinearSegmentedColormap.from_list("", [colours_map["J"], "white", colours_map["P"]])

# labels
goehring_labels = ["aPar", "pPar"]
par3add_labels = ["$J$ (Par3)", "$M$ (Par3-Par6-PKC)",
                  "$A$ (CDC42-Par6-PKC)", "$P$ (Posterior)"]

xlabel = "$z$" #"$x$ (μm)"
ylabel = "$Y_i$" #r"$\text{μm}^{-2}$"

# Change the default plot settings for all plots
plt.rcParams["axes.xmargin"] = 0
plt.rc('font', size=font_size)
plt.rc('xtick', labelsize=label_font_size)
plt.rc('ytick', labelsize=label_font_size)
plt.rc('lines', linewidth=line_width)
plt.rc('legend', fontsize=label_font_size)



# Convert a single time point from the ODE solver to a pandas dataframe for easier plotting ------------------------------------
def convert_output_to_pandas(X, Y, species = ("J", "M", "A", "P")):
    Nx = len(X)
    assert(np.shape(Y)[0] == (Nx*len(species)))
    df = DataFrame({'x': X})
    for i, sp in enumerate(species):
        idx0 = i*Nx
        df[sp] = Y[idx0:(idx0+Nx)]
    return df


# Convert a multiple time points from the ODE solver to a pandas dataframe for easier plotting ------------------------------------
def convert_time_output_to_pandas(X, Y, t, species = ("J", "M", "A", "P")):
    df_list = [convert_output_to_pandas(X,Y[:,i], species=species).assign(t = t_i) for i,t_i in enumerate(t)]
    return concat(df_list, ignore_index=True)



# Plot the species profiles --------------------------------------------------------------------------------------------------
def plot_time_point(ax, X, Y, legend = False, v_func = None, species = ("J", "M", "A", "P")):

    df = convert_output_to_pandas(X, Y, species=species)
    X = df['x']
    for sp in species:
        ax.plot(X, df[sp], label=f"${sp}$", color = colours_map[sp])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if v_func is not None:
        v = [v_func(xi)/0.0015 for xi in X]
        ax.plot(X, v, label="$\\dfrac{\\tilde{v}}{\\tilde{v}_{\\text{max}}}$", color = "black", linestyle = "--", linewidth = 1)

    if legend:
        ax.legend(fontsize=label_font_size)


# Function to add a shared legend to a figure with multiple subplots ---------------------------------------------------
def add_shared_legend(fig, axs):
    handles, labels = [], []
    for ax in axs:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc='outside right', fontsize=label_font_size)


# Format parameter labels for plotting using math mode ---------------------------------------------------------------
def format_param_label_math(label: str) -> str:
    s = str(label)
    if s.startswith("k") and "_" not in s:
        s = f"k_{s[1:]}"
    if s.startswith("rho"):
        s = "\\" + s
    if s.startswith("k_"):
        base_part, sub_part = s.split("_", 1)
        if sub_part.startswith("off"):
            sub_part = "off," + sub_part[3:]
        elif sub_part.startswith("on"):
            sub_part = "on," + sub_part[2:]
        elif sub_part.startswith("dis"):
            sub_part = "dis," + sub_part[3:]
        s = f"{base_part}_{sub_part}"
    if "_" in s:
        base, sub = s.split("_", 1)
        return f"${base}_{{{sub}}}$"
    return s



def animate_plot(sol, kvals, save_file:str = None, rescale=False):

    # rescale so maximal protein quantity is 1
    v_rescale_for_visibility = 1.0/(0.0015*kvals.xL)

    # Initial plot
    fig, ax = plt.subplots()
    lines = []
    df = convert_output_to_pandas(kvals.X, sol.y[:, 0], species=kvals.Species)
    for i, sp in enumerate(kvals.Species):
        line, = ax.plot(kvals.X, df[sp], label=sp, color=colours_map[sp])
        lines.append(line)

    # p_m = polarity_measure(kvals.X, sol.y[:, 0], model)
    # time_label = ax.text(0.1, 1.05, f"t={sol.t[0]} p={p_m:.4f}", transform=ax.transAxes, ha="center")
    time_label = ax.text(0.1, 1.05, f"t={sol.t[0]}", transform=ax.transAxes, ha="center")
    linev, = ax.plot(kvals.X, [v_rescale_for_visibility*kvals.v_func(kvals, x, 0) for x in kvals.X], label="v", linestyle="--", color="black")
    # ax.text(0.7, 1.05, kvals.label + ";Nx:" + str(Nx), transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals.x0, kvals.xL], ylim=[np.min(sol.y)-0.05,np.max(sol.y)+0.05], xlabel="x", ylabel="Y")
    ax.legend()

    def animate(t_i):
        df_i = convert_output_to_pandas(kvals.X, sol.y[:, t_i], species=kvals.Species)
        linev.set_ydata([v_rescale_for_visibility*kvals.v_func(kvals, x, sol.t[t_i]) for x in kvals.X])
        for i, sp in enumerate(kvals.Species):
            lines[i].set_ydata(df_i[sp])
        # p_m = polarity_measure(kvals.X, sol.y[:, t_i], model)
        time_label.set_text(f"t={sol.t[t_i]:.2f}")# p={p_m:.4f}")
        return (*lines, linev, time_label)

    ani = animation.FuncAnimation(fig, animate, interval=10000/len(sol.t), blit=True, frames=len(sol.t))

    if save_file is not None:
        file_name = FIGURES_DIR / f"{save_file}.gif"
        print(f"Saving animation to {file_name}")
        ani.save(file_name)

    plt.show(block=False)


# def plot_final_timestep(sol, kvals, rescale=False):
#     plt.figure()
#     ax = plt.subplot()

#     Nx = kvals.Nx
#     scalar = 1 if not rescale else np.max(sol.y)

#     ax.plot(kvals.X, sol.y[:Nx, -1] / scalar, label="par3", color="green")
#     ax.plot(kvals.X, sol.y[Nx:2 * Nx, -1] / scalar, label="par3-PKC", color="purple")
#     ax.plot(kvals.X, sol.y[2 * Nx:3 * Nx, -1] / scalar, label="cdc42-PKC", color="blue")
#     ax.plot(kvals.X, sol.y[3 * Nx:, -1] / scalar, label="posterior", color="orange")

#     p_m, _, _ = polarity_get_all(kvals.X, sol.y[2*Nx:3*Nx, -1], sol.y[3*Nx:, -1], Nx)
#     ax.text(0.1, 1.05, f"t={sol.t[-1]},p={p_m:.4f}", transform=ax.transAxes, ha="center")  # time value
#     ax.plot(kvals.X, [kvals.v_func(kvals, x, sol.t[-1]) for x in kvals.X], label="v", linestyle="--", color="black")  # v_func

#     ax.text(0.7, 1.05, kvals.label, transform=ax.transAxes, ha="center")

#     # ax.set(xlim=[kvals.x0, kvals.xL], ylim=[np.min(sol.y[:, -1])/scalar-0.05, np.max(sol.y[:, -1])/scalar+0.05], xlabel="x", ylabel="A/P")
#     ax.legend()

#     plt.show(block=False)


# # polarity based on the cdc42 quantity for Anterior
# # untested code
# def plot_variation_sets(variation_sets, label=DEFAULT_PARAMETERS["label"], x_axis_labels: list[str] | None = None, show_orientation=True, xlim=None):
#     plt.figure()
#     ax = plt.subplot()

#     # add then remove plot with xticks so that they get ordered correctly in the figure
#     sentinel, = ax.plot(x_axis_labels, [0.5]*len(x_axis_labels))
#     sentinel.remove()

#     for i in np.arange(0, len(variation_sets)):
#         variation = variation_sets[i]
#         sol_list = variation[0]
#         kvals_list = variation[1]

#         polarity_m_list = []
#         xticks = []
#         if len(variation_sets) > 7:
#             color = (np.minimum(1, 0.3 + (i % 6)/7), 0.75 - 0.50*i/len(variation_sets),0.5 + 0.50*i/len(variation_sets))
#         else:
#             color = (np.minimum(1, 0.3 + (i % 3)/4), 0.75 - 0.50*i/len(variation_sets),0.5 + 0.50*i/len(variation_sets))

#         for j in np.arange(0, len(sol_list)):
#             sol = sol_list[j]
#             kvals = kvals_list[j]

#             if not sol == "FAILURE":
#                 p_measure, _p_orientation, p_marker = polarity_get_all(kvals.X, sol.y[2*kvals.Nx:3*kvals.Nx, -1], sol.y[3*kvals.Nx:, -1], kvals.Nx)

#                 xtick = x_axis_labels[j] if x_axis_labels is not None else j
#                 marker = 'o' if not show_orientation else p_marker

#                 # jitter the near-0 values so they are visible
#                 if p_measure<0.02:
#                     p_measure += 0.02*i/len(variation_sets)-0.01

#                 polarity_m_list.append(p_measure)
#                 xticks.append(xtick)
#                 ax.scatter(xtick, p_measure, color=color, marker=marker, s=100)

#         ax.plot(xticks, polarity_m_list, "--", label=kvals_list[1]["key_varied"], color=color)

#     ax.legend()
#     ax.set(xlabel="percentage of baseline value", ylabel="polarity", ylim=[-0.1,1.1], xlim=xlim)
#     ax.title.set_text(label)
#     ax.tick_params(which="both", labelsize=15)
#     ax.tick_params(axis='x', labelrotation=60)
#     plt.show(block=False)


# # plot combined A,M,J (aPars)

# def animate_plot_apar_combo(sol, kvals: dict, save_file=False, file_code: str = None, rescale=False, no_par3=False):
#     if file_code is None:
#         file_code = f'{time.time_ns()}'[5:]

#     Nx = kvals.Nx
#     # J = U[:Nx]
#     # M = U[Nx:2 * Nx]
#     # A = U[2 * Nx:3 * Nx]
#     # P = U[3 * Nx:]

#     combined_apar = []

#     for i in np.arange(0,len(sol.t)):
#         combined_apar.append((sol.y[Nx:2*Nx, i] + sol.y[2*Nx:3*Nx, i]) if no_par3 else (sol.y[:Nx, i]) + sol.y[Nx:2*Nx, i] + sol.y[2*Nx:3*Nx, i])
#     scalar = 1 if not rescale else np.max(sol.y)
#     v_rescale_for_visibility = np.maximum(np.max(sol.y), np.max(combined_apar[0]))/scalar * 10  # rescale so 0.1 is equal to max protein quantity in the plotting of v

#     fig, ax = plt.subplots()
#     line1, = ax.plot(kvals.X, combined_apar[0]/scalar, label="anterior", color="green")
#     line2, = ax.plot(kvals.X, sol.y[3*Nx:, 0]/scalar, label="posterior", color="orange")
#     p_m, _, _ = polarity_get_all(kvals.X, sol.y[2*Nx:3*Nx, 0], sol.y[3*Nx:, 0], Nx)  # polarisation metric
#     time_label = ax.text(0.1, 1.05, f"t={sol.t[0]} p={p_m:.4f}", transform=ax.transAxes, ha="center")
#     linev, = ax.plot(kvals.X, [v_rescale_for_visibility*kvals.v_func(kvals, x, 0) for x in kvals.X], label="v", linestyle="--", color="black")

#     ax.text(0.7, 1.05, kvals.label + ";Nx:" + str(Nx), transform=ax.transAxes, ha="center")

#     maxy = np.maximum(np.max(sol.y), np.max(combined_apar));

#     ax.set(xlim=[kvals.x0, kvals.xL], ylim=[np.min(sol.y)/scalar-0.05,maxy/scalar+0.05], xlabel="x", ylabel="par3,A/P")
#     ax.legend()
#     ax.set_title("apar combo, plot without par3" if no_par3 else "apar combo")

#     def animate(t_i):
#         linev.set_ydata([v_rescale_for_visibility*kvals.v_func(kvals, x, sol.t[t_i]) for x in kvals.X])
#         line1.set_ydata(combined_apar[t_i]/scalar)
#         line2.set_ydata(sol.y[3*Nx:, t_i]/scalar)
#         p_m, _, _ = polarity_get_all(kvals.X, sol.y[2*Nx:3*Nx, t_i], sol.y[3*Nx:, t_i], Nx)
#         time_label.set_text(f"t={sol.t[t_i]:.2f} p={p_m:.4f}")
#         return (line1, line2, linev, time_label)

#     ani = animation.FuncAnimation(fig, animate, interval=10000/len(sol.t), blit=True, frames=len(sol.t))

#     if save_file:
#         file_name = FIGURES_DIR / f"{file_code}_spatialPar.gif"
#         print(f"Saving animation to {file_name}")
#         ani.save(file_name)

#     plt.show(block=False)


# def plot_final_timestep_apar_combo(sol, kvals):
#     plt.figure()
#     ax = plt.subplot()

#     Nx = kvals.Nx
#     scalar = 1

#     combined_apar = (sol.y[:Nx, -1] + sol.y[Nx:2*Nx, -1] + sol.y[2*Nx:3*Nx, -1])

#     ax.plot(kvals.X, combined_apar / scalar, label="anterior", color="green")
#     ax.plot(kvals.X, sol.y[3 * Nx:, -1] / scalar, label="posterior", color="orange")

#     p_m, _, _ = polarity_get_all(kvals.X, sol.y[2*Nx:3*Nx, -1], sol.y[3*Nx:, -1], Nx)
#     ax.text(0.1, 1.05, f"t={sol.t[-1]},p={p_m:.4f}", transform=ax.transAxes, ha="center")  # time value
#     ax.plot(kvals.X, [kvals.v_func(kvals, x, sol.t[-1]) for x in kvals.X], label="v", linestyle="--", color="black")  # v_func

#     ax.text(0.7, 1.05, kvals.label, transform=ax.transAxes, ha="center")
#     ax.set_title("apar combo")

#     ax.legend()

#     plt.show(block=False)


#     p_m, _, _ = polarity_get_all(kvals.X, sol.y[2*Nx:3*Nx, -1], sol.y[3*Nx:, -1], Nx)
#     ax.text(0.1, 1.05, f"t={sol.t[-1]},p={p_m:.4f}", transform=ax.transAxes, ha="center")  # time value
#     ax.plot(kvals.X, [kvals.v_func(kvals, x, sol.t[-1]) for x in kvals.X], label="v", linestyle="--", color="black")  # v_func

#     ax.text(0.7, 1.05, kvals.label, transform=ax.transAxes, ha="center")

#     # ax.set(xlim=[kvals.x0, kvals.xL], ylim=[np.min(sol.y[:, -1])/scalar-0.05, np.max(sol.y[:, -1])/scalar+0.05], xlabel="x", ylabel="A/P")
#     ax.legend()

#     plt.show(block=False)



# # plot cytoplasmic quantities over time
# def plot_cyto(sol, kvals):
#     plt.figure()
#     ax = plt.subplot()

#     ax.plot(sol.t, [kvals.A_cyto(kvals, sol.y[:kvals.Nx, t_i]) for t_i in np.arange(0, len(sol.t))], label="A_cyto", color="blue")
#     ax.plot(sol.t, [kvals.P_cyto(kvals, sol.y[kvals.Nx:, t_i]) for t_i in np.arange(0, len(sol.t))], label="P_cyto", color="orange")

#     ax.text(1, 1.05, kvals.label, transform=ax.transAxes, ha="center")

#     ax.set(xlabel="time")

#     ax.title.set_text("Cytoplasmic Quantities")

#     ax.legend()
#     plt.show(block=False)

# def plot_overall_quantities_over_time(sol, kvals, rescale_by_length=True):
#     plt.figure()
#     ax = plt.subplot()

#     # since this is overall quantity, rescale by space length
#     length_scalar = 1 if not rescale_by_length else np.abs(kvals.xL - kvals.x0)

#     #TODO - unsure if I should plot with or without the psi multiple
#     ax.plot(sol.t, [kvals.A_cyto(kvals, sol.y[:kvals.Nx, t_i])/length_scalar for t_i in np.arange(0, len(sol.t))],
#             label="A_cyto", color="blue", linestyle="--")
#     ax.plot(sol.t, [kvals.P_cyto(kvals, sol.y[kvals.Nx:, t_i])/length_scalar for t_i in np.arange(0, len(sol.t))],
#             label="P_cyto", color="orange", linestyle="--")

#     ax.plot(sol.t, [Ybar(kvals, sol.y[:kvals.Nx, t_i])/length_scalar for t_i in np.arange(0, len(sol.t))], label="A_bar", color="blue")
#     ax.plot(sol.t, [Ybar(kvals, sol.y[kvals.Nx:, t_i])/length_scalar for t_i in np.arange(0, len(sol.t))], label="P_bar", color="orange")

#     ax.text(1, 1.05, kvals.label, transform=ax.transAxes, ha="center")

#     ax.set(xlabel="time")

#     ax.title.set_text("Quantities")

#     ax.legend()
#     plt.show(block=False)


# # plot a bunch of different solutions final timestep (just A,P) on single figure
# # Assumes that all solutions have the same X,Nx,x0,xL, and time points
# def plot_multi_final_timestep(sol_list, kvals_list, label=DEFAULT_PARAMETERS["label"], plot_A=True, plot_P=True):
#     kvals = kvals_list[0]

#     plt.figure()
#     ax = plt.subplot()

#     for i in np.arange(0,len(sol_list)):
#         sol = sol_list[i]
#         kvals_this_sol = kvals_list[i]

#         if plot_A:
#             ax.plot(kvals.X, sol.y[:kvals.Nx, -1], label=f"A_{kvals_this_sol['label']}", color=(0.3 + (i % 3)/4, 0.75 - 0.50*i/len(sol_list),0.5 + 0.50*i/len(sol_list)))
#         if plot_P:
#             ax.plot(kvals.X, sol.y[kvals.Nx:, -1], label=f"P_{kvals_this_sol['label']}", color=(0.3 + (i % 3)/4, 0.75 - 0.50*i/len(sol_list),0.5 + 0.50*i/len(sol_list)))

#     ax.text(0.1, 1.05, f"t={sol_list[0].t[-1]}", transform=ax.transAxes, ha="center") # timestamp
#     ax.text(1, 1.05, label, transform=ax.transAxes, ha="center") # label

#     ax.set(xlim=[kvals.x0, kvals.xL], ylim=[np.min([sol.y[:, -1] for sol in sol_list])-0.05, np.max([sol.y[:, -1] for sol in sol_list])+0.05], xlabel="x", ylabel="A/P")
#     ax.title.set_text("Multiple Sims")
#     ax.legend()

#     plt.show(block=False)

# def plot_failure(U, t, kvals):
#     plt.figure()
#     ax = plt.subplot()

#     ax.plot(kvals.X, U[:kvals.Nx], label="anterior", color="blue")
#     ax.plot(kvals.X, U[kvals.Nx:], label="posterior", color="orange")
#     ax.text(0.1, 1.05, f"t={t}", transform=ax.transAxes, ha="center")
#     ax.plot(kvals.X, [kvals.v_func(kvals, x, t) for x in kvals.X], label="v", linestyle="--", color="black")

#     ax.text(1, 1.05, kvals.label, transform=ax.transAxes, ha="center")

#     ax.set(xlim=[kvals.x0, kvals.xL], ylim=[np.min(U)-0.05, np.max(U)+0.05], xlabel="x", ylabel="A/P")
#     ax.title.set_text("Failure Plot")
#     ax.legend()

#     plt.show(block=True)


# # assume lists are [base, ...others]
# def plot_metric_comparisons(sol_list, kvals_list, label=DEFAULT_PARAMETERS["label"]):
#     assert len(sol_list) == len(kvals_list)

#     # kvals = kvals_list[0]
#     # polarity measure metric (final timestep)
#     # plt.figure()
#     # ax1 = plt.subplot()

#     # TODO other metric
#     # plt.figure()
#     # ax2 = plt.subplot()

#     plt.figure()

#     for i in np.arange(0, len(sol_list)):
#         sol = sol_list[i]
#         kvals = kvals_list[i]

#         polarity_m = polarity_measure(kvals.X, sol.y[:kvals.Nx, -1], sol.y[kvals.Nx:, -1], kvals.Nx)

#         plt.plot(0, polarity_m, marker="o", linestyle="None", label=kvals.label)

#     plt.legend()
#     # plt.title.set_text(label)
#     plt.show(block=False)


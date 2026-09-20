# Universal thing for figures
# ordering for par3add elements is [J, M, A, P]
# for goehring the ordering is just [A, P]
from pathlib import Path
import numpy as np
from pandas import DataFrame, concat
from matplotlib import pyplot as plt, animation
from dataclasses import is_dataclass, asdict

from polarity.model_enums import MODELS

FIGURES_DIR = Path(__file__).resolve().parents[3] / "figures"
DATA_DIR = Path(__file__).resolve().parents[3] / "results"

# Plot style
font_size = 16
label_font_size = 14
line_width = 3
page_width_fig = 12
# Set global savefig default to tight bounding box
plt.rcParams['savefig.bbox'] = 'tight'

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
    assert np.shape(Y)[0] == (Nx*len(species)), f"Y shape {np.shape(Y)} does not match expected shape {(Nx*len(species),)}"
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



def animate_plot(sol, kvals: dict, save_file = None):

    # rescale so maximal protein quantity is 1
    v_rescale_for_visibility = 1.0/(0.0015*kvals["xL"])

    # Initial plot
    fig, ax = plt.subplots()
    lines = []
    df = convert_output_to_pandas(kvals["X"], sol.y[:, 0], species=kvals["Species"])
    for i, sp in enumerate(kvals["Species"]):
        line, = ax.plot(kvals["X"], df[sp], label=sp, color=colours_map[sp])
        lines.append(line)

    # p_m = polarity_measure(kvals["X"], sol.y[:, 0], model)
    # time_label = ax.text(0.1, 1.05, f"t={sol.t[0]} p={p_m:.4f}", transform=ax.transAxes, ha="center")
    time_label = ax.text(0.1, 1.05, f"t={sol.t[0]}", transform=ax.transAxes, ha="center")
    linev, = ax.plot(kvals["X"], [v_rescale_for_visibility*kvals["v_func"](kvals, x, 0) for x in kvals["X"]], label="v", linestyle="--", color="black")
    # ax.text(0.7, 1.05, kvals["label"] + ";Nx:" + str(kvals["Nx"]), transform=ax.transAxes, ha="center")

    ax.set(xlim=[kvals["x0"], kvals["xL"]], ylim=[np.min(sol.y)-0.05,np.max(sol.y)+0.05], xlabel="x", ylabel="Y")
    ax.legend()

    def animate(t_i):
        df_i = convert_output_to_pandas(kvals["X"], sol.y[:, t_i], species=kvals["Species"])
        linev.set_ydata([v_rescale_for_visibility*kvals["v_func"](kvals, x, sol.t[t_i]) for x in kvals["X"]])
        for i, sp in enumerate(kvals["Species"]):
            lines[i].set_ydata(df_i[sp])
        # p_m = polarity_measure(kvals["X"], sol.y[:, t_i], model)
        time_label.set_text(f"t={sol.t[t_i]:.2f}")# p={p_m:.4f}")
        return (*lines, linev, time_label)

    ani = animation.FuncAnimation(fig, animate, interval=10000/len(sol.t), blit=True, frames=len(sol.t))

    if save_file is not None:
        file_name = FIGURES_DIR / f"{save_file}.gif"
        print(f"Saving animation to {file_name}")
        ani.save(file_name)

    plt.show(block=False)

# Discover parameters by iterative grid search
# Output is grep-able to get list of parameter changes
# search for 'Best point was'

import time
import numpy as np
from numpy import linalg
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
from pickle import dump, load

from polarity.utilities import model_task_handler, figure_helper as fh
from polarity.model_enums import MODELS, model_to_module

FORCE_RERUN = False  # if True, will re-run all tasks even if results are already saved

MODULE_PAR3ADD = model_to_module(MODELS.PAR3ADD)
MODULE_GOEHRING = model_to_module(MODELS.GOEHRING)

params_goehring = MODULE_GOEHRING.DEFAULT_PARAMETERS

# starting parameters
base_params_par3add = {"psi": params_goehring.psi,
                       "D_J": params_goehring.D_A,
                       "D_M": params_goehring.D_P/2,
                       "D_A": params_goehring.D_A,
                       "D_P": params_goehring.D_P,

                       "kJP": 0.08,
                       "kMP": 0.07,
                       "kAP": params_goehring.kAP/2,
                       "kPA": params_goehring.kPA,

                       "konJ": 0.014,
                       "konA": 0,  # not used in writeup
                       "konP": params_goehring.konP,

                       "koffJ": params_goehring.koffA/2,
                       "koffM": params_goehring.koffA,
                       "koffA": params_goehring.koffA,
                       "koffP": params_goehring.koffP,

                       "konM": params_goehring.konA,
                       "kdisp": 0.0022,

                       "rho_J": 1.2,
                       "rho_A": params_goehring.rho_A,
                       "rho_P": params_goehring.rho_P,

                       "sigmaJ": 1, "sigmaM": 1, "sigmaP": 1,
                       "alpha": 1, "beta": 2,
                       }

# parameters we will vary
PARAMS_TO_VARY = ["kJP", "kMP", "kAP", "konJ",
                  "koffJ", "koffM", "koffA",
                  "konM", "kdisp",
                  "rho_J"
                  ]

# PARAMS_TO_VARY = ["kJP", "kMP", "kAP"]

# set range of parameter variation
MAX_MULTIPLIER = 1.25
MIN_MULTIPLIER = 0.75
TOTAL_STEPS = 4

NX = 100  # spatial discretisation
TL_HOM = 3000  # end time to get homogeneous steadys-state
TL_EST = 9000  # end time for establishment

# initial condition for par3add to get a-dominant homogeneous steady-state
INIT_COND_HOM = [1]*NX + [1]*(2*NX) + [0]*NX

LABEL_P_HOM = "par3add_hom_run"
LABEL_P_POL = "par3add_pol_run"

FIG_OUTPUT_FOLDER = Path( fh.FIGURES_DIR / "par3add_parameter_search" )
DAT_OUTPUT_FOLDER = Path( fh.DATA_DIR / "par3add_parameter_search")
# Create output folders if they don't exist
if not FIG_OUTPUT_FOLDER.exists():
    FIG_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
if not DAT_OUTPUT_FOLDER.exists():
    DAT_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

NO_PLOT = True
MAX_REPETITIONS = 27  # max iterations of full variations run-through


def v_func_zero(kvals, x, t):
    return 0


def main():

    # Generate parameter variation pairs
    variation_multipliers = [MIN_MULTIPLIER+i*(MAX_MULTIPLIER-MIN_MULTIPLIER)/(TOTAL_STEPS-1) for i in range(TOTAL_STEPS)]

    variation_pairs = []

    for i in range(len(PARAMS_TO_VARY)):
        for j in range(i+1, len(PARAMS_TO_VARY)):
            variation_pairs.append((PARAMS_TO_VARY[i], PARAMS_TO_VARY[j]))

    print(variation_pairs)

    # Get goehring runs for comparison
    print("Getting goehring results")
    goehring_res = get_goehring_res()

    # Run parameter variations
    print("Beginning variations")

    params_par3add = base_params_par3add

    for rep_i in range(MAX_REPETITIONS):  # Iterate multiple consecutive runs
        print(f"Running run number {rep_i+1}/{MAX_REPETITIONS}")

        best_point = do_variations(params_par3add, goehring_res, variation_pairs, variation_multipliers, rep_i)

        print(f"Finished run {rep_i+1}/{MAX_REPETITIONS}")
        print(f"Best point was {best_point}")

        params_par3add = {**params_par3add, **best_point[0]}

        if best_point is None:
            break

    plt.show()

    # Output best point to file
    with open( DAT_OUTPUT_FOLDER / "best_point.pkl", 'wb') as f:
        final_params = {**params_par3add, **best_point[0]}
        dump(final_params, f)
    # Also print to console
    print("\033[1;32mFinal parameter set:\033[0m")
    for key, value in final_params.items():
        print(f"{key}: {value:.4f}")


def do_variations(params_par3add, goehring_results: tuple[list], variation_pairs: list[tuple],
                  variation_multipliers: tuple[dict, float], iter_id: int) -> tuple[dict, float]:

    best_point: tuple[dict, float] = ({}, None)

    for i_variation_pair in range(len(variation_pairs)):
        variation_pair = variation_pairs[i_variation_pair]

        # generate variation sets
        p1 = variation_pair[0]
        p2 = variation_pair[1]

        varied_params = {p1: params_par3add[p1], p2: params_par3add[p2]}

        print(f"{i_variation_pair+1}/{len(variation_pairs)} - Working on {varied_params}")

        param_pairs = []

        # create all parameter combos
        for variation1 in variation_multipliers:
            for variation2 in variation_multipliers:
                param_pairs.append((variation1*varied_params[p1], variation2*varied_params[p2]))

        # generate tasks
        tasks_hom = []
        for ppair in param_pairs:
            task = (MODELS.PAR3ADD, {**params_par3add, p1: ppair[0], p2: ppair[1],
                                     "Nx": NX, "tL": TL_HOM, "initial_condition": INIT_COND_HOM,
                                     "label": f"{LABEL_P_HOM}_{p1}_{ppair[0]:.4f}_{p2}_{ppair[1]:.4f}",
                                     "v_func": v_func_zero,
                                     })
            tasks_hom.append(task)

        # load or run to get results
        res_hom_all = load_or_run(f"{LABEL_P_HOM}_{p1}_{p2}_{iter_id}", tasks_hom)
        for label, res in res_hom_all:
            res[1]["variation_info"] = {p1: res[1][p1], p2: res[1][p2]}  # add variation info to results for later reference

        # generate polarisation tasks
        tasks_pol = []
        for label, res in res_hom_all:
            if res[1] != "FAILURE":
                varied_params_value = res[1]["variation_info"]
                init_cond_pol = res[0].y[:, -1]

                # takes long time, so skipping stuff that's way off
                hsim = calculate_similarity_gp(goehring_results[0], init_cond_pol)
                if hsim > 5:
                    continue
                task = (MODELS.PAR3ADD, {**params_par3add,
                                         **varied_params_value,
                                         "Nx": NX, "tL": TL_EST, 
                                         "initial_condition": init_cond_pol,
                                         "label": f"{LABEL_P_POL}_{p1}_{varied_params_value[p1]:.4f}_{p2}_{varied_params_value[p2]:.4f}"
                                         })
                tasks_pol.append(task)
            else:
                print("failure detected")

        # load or run to get results
        res_pol_all = load_or_run(f"{LABEL_P_POL}_{p1}_{p2}_{iter_id}", tasks_pol)
        for label, res in res_pol_all:
            res[1]["variation_info"] = {p1: res[1][p1], p2: res[1][p2]}

        # compare with goehring
        comparisons = goehring_comparer(goehring_results, res_hom_all, res_pol_all)
        for c in comparisons:
            if best_point[1] is None or best_point[1] > c[2]:  # found lower polarised difference
                best_point = (c[0], c[2])

        # plot
        if not NO_PLOT:
            print("\n".join([f"{c}" for c in comparisons]))

            # plot for homogeneous comparison
            fig = plot_gcomparisons(p1, p2, [(c[0], c[1]) for c in comparisons], (params_par3add[p1], params_par3add[p2]))
            # save plot
            fig_file_name = FIG_OUTPUT_FOLDER/ f"{f'{time.time_ns()}'[5:]}{LABEL_P_HOM}_compare_{p1}_{p2}"
            fig.savefig(fig_file_name)
            print(f"Saved figure to {fig_file_name}")

            # plot for polarised comparison
            fig = plot_gcomparisons(p1, p2, [(c[0], c[2]) for c in comparisons], (params_par3add[p1], params_par3add[p2]))
            # save plot
            fig_file_name = FIG_OUTPUT_FOLDER/ f"{f'{time.time_ns()}'[5:]}{LABEL_P_POL}_compare_{p1}_{p2}"
            fig.savefig(fig_file_name)
            print(f"Saved figure to {fig_file_name}")
            plt.close("all")

    # animate plot for best detected parameter pair
    if True and best_point is not None:
        for label, res in res_pol_all:
            if res[0] != "FAILURE" and res[1]["variation_info"] == best_point[0]:
                print("Animating plot of best point")
                filename = "best_point-" + "-".join(f"{key}_{value:.5f}" for key, value in best_point[0].items())
                print(filename)
                fh.animate_plot(res[0], res[1], save_file = FIG_OUTPUT_FOLDER / filename )

                break

    return best_point


def plot_gcomparisons(p1, p2, comparisons: list[tuple[dict, float]], baseline_point) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))

    max_val = np.max([[v[1]] for v in comparisons])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["violet", "blue", "yellow"])
    norm = plt.Normalize(0, max_val)

    for point in comparisons:
        v1 = point[0][p1]
        v2 = point[0][p2]
        comp_val = point[1]
        ax.scatter(v1, v2, c=comp_val, cmap=cmap, norm=norm, marker="o", s=100)

    sc = ax.scatter(baseline_point[0], baseline_point[1], c=max_val, cmap=cmap,
                norm=norm, marker=".", s=100)  # plot existing params point for comparison

    clb = fig.colorbar(sc, ax=ax)
    clb.ax.set_title(f"g_diff max:{max_val:.2f}")

    ax.set_xlabel(p1)
    ax.set_ylabel(p2)
    ax.set_title(f"{p1}, {p2}")
    return fig


def goehring_comparer(goehring_results: tuple[list], res_hom_all: list[tuple],
                      res_pol_all: list[tuple]) \
                        -> list[tuple[dict, float, float]]:
    assert len(res_pol_all) <= len(res_hom_all)

    out_list: list[tuple[dict, float, float]] = []

    for label, res in res_pol_all:
        if res[0] != "FAILURE":
            varied_param_values: dict = res[1]["variation_info"]

            goehring_sim_hom = None

            # find equivalent result from homogeneous run
            for label, hom_res in res_hom_all:
                if hom_res[0] != "FAILURE":
                    hom_variation_info = hom_res[1]["variation_info"]
                    if hom_variation_info == varied_param_values:
                        goehring_sim_hom = calculate_similarity_gp(goehring_results[0], hom_res[0].y[:, -1])

            # calculate goehring simularity (polarised)
            goehring_sim_pol = calculate_similarity_gp(goehring_results[1], res[0].y[:, -1])

            # calculate goehring simularity during establishment
            full_g_res_multi = [
                                (get_goehring_res(pol_time_divider=10)[1], 10),
                                (get_goehring_res(pol_time_divider=5)[1], 5),
                                (get_goehring_res(pol_time_divider=4)[1], 4),
                                (get_goehring_res(pol_time_divider=3)[1], 3),
                                (get_goehring_res(pol_time_divider=2)[1], 2),
                                (get_goehring_res(pol_time_divider=1.5)[1], 1.5),
                                ]

            goehring_sim_partway_all = []

            for full_g_res_tuple in full_g_res_multi:
                goehring_sim_partway = calculate_similarity_gp(full_g_res_tuple[0], res[0].y[:, int(res[0].y.shape[1]//full_g_res_tuple[1])])
                goehring_sim_partway_all.append(goehring_sim_partway)

            goehring_sim_partway = sum(goehring_sim_partway_all)//len(full_g_res_multi)

            if goehring_sim_hom is not None:
                out_list.append((varied_param_values, goehring_sim_hom, goehring_sim_pol+goehring_sim_partway/2))
        else:
            print("failure detected")

    return out_list


def calculate_similarity_gp(goehring_res: list, par3add_res: list):
    return linalg.vector_norm(goehring_res -
                              np.concatenate((par3add_res[NX:2*NX] + par3add_res[2*NX:3*NX], par3add_res[3*NX:])))


def get_goehring_res(pol_time_divider: int = 1) -> tuple:
    filename = DAT_OUTPUT_FOLDER / f"goehring_results.pkl"
    if filename.exists():
        try:
            with open(filename, 'rb') as f:
                loaded = load(f)
            return loaded
        except Exception as e:
            print(f"Failed to load goehring results from {filename}: {e}")
    else:
        print("Running goehring model to get results for comparison")
        res_h = get_goehring_homo_ic()
        res_p = get_goehring_polarised(res_h, pol_time_divider)
        with open(filename, 'wb') as f:
            dump((res_h, res_p), f)
        return (res_h, res_p)


def get_goehring_homo_ic():
    task = (MODELS.GOEHRING, {"tL": TL_HOM,
                              "initial_condition": [1]*NX + [0]*NX, "v_func": v_func_zero})
    label, res = model_task_handler.run_tasks([task])[0]
    return res[0].y[:, -1]


def get_goehring_polarised(initial_condition, pol_time_divider: int = 1):
    task = (MODELS.GOEHRING, {"tL": TL_EST,
                              "initial_condition": initial_condition})
    label, res = model_task_handler.run_tasks([task])[0]
    # return res[1].y[:, -1] if not pol_half_time else res[1].y[:, res[1].y.shape(1)//pol_time_divider]
    return res[0].y[:, (res[0].y.shape[1]-1)//pol_time_divider]


def load_or_run(name: str, tasks: list[tuple]) -> list[tuple]:
    filename = DAT_OUTPUT_FOLDER / f"{name}.pkl"

    # Load the data if it exists
    if filename.exists() and not FORCE_RERUN:
        try:
            with open(filename, 'rb') as f:
                loaded_data = load(f)
            print(f"Loading of {name} succeeded!")
            return loaded_data
        except Exception as e:
            print(f"Failed loading of {name} because: " + str(e))
    # Otherwise run and save the data
    else:
        res = model_task_handler.run_tasks_parallel(tasks, NUMBER_OF_PROCESSES=12)
        print(f"Saving results {name}")
        with open(filename, 'wb') as f:
            dump(res, f)
        return res


# def bad_hash_for_filename(tasks):
#     bad_hash = 0

#     for task in tasks:
#         params = task[1]
#         for key in params:
#             element = params[key]
#             if isinstance(element, (int, float)):
#                 bad_hash += element
#             elif key == "initial_condition":
#                 bad_hash += np.sum(element)
#     bad_hash += len(tasks)
#     return bad_hash


if __name__ == '__main__':
    matplotlib.use('Agg')  # block plots from appearing
    main()

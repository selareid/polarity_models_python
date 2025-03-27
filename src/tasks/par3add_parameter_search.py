# Discover parameters by iterative grid search
# Output is grep-able to get list of parameter changes
# search for 'Best point was'

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import time
import numpy as np
from numpy import linalg
from ..models import MODELS, model_to_module
import matplotlib
from matplotlib import pyplot as plt
from src import model_task_handler


MODULE_PAR3ADD = model_to_module(MODELS.PAR3ADD)

params_goehring = {"psi": 0.174,
                   "D_A": 0.28, "D_P": 0.15,
                   "k_onA": 8.58 * 10 ** (-3), "k_onP": 4.74 * 10 ** (-2),
                   "k_offA": 5.4 * 10 ** (-3), "k_offP": 7.3 * 10 ** (-3),
                   "k_AP": 0.190, "k_PA": 2.0,
                   "rho_A": 1.56, "rho_P": 1.0,
                   "alpha": 1, "beta": 2,
                   }

# starting parameters
base_params_par3add = {"psi": params_goehring["psi"],
                       "D_J": params_goehring["D_A"],
                       "D_M": params_goehring["D_P"]/2,
                       "D_A": params_goehring["D_A"],
                       "D_P": params_goehring["D_P"],

                       "kJP": 0.08,
                       "kMP": 0.07,
                       "kAP": params_goehring["k_AP"]/2,
                       "kPA": params_goehring["k_PA"],

                       "konJ": 0.014,
                       "konA": 0,  # not used in writeup
                       "konP": params_goehring["k_onP"],

                       "koffJ": params_goehring["k_offA"]/2,
                       "koffM": params_goehring["k_offA"],
                       "koffA": params_goehring["k_offA"],
                       "koffP": params_goehring["k_offP"],

                       "k1": params_goehring["k_onA"],
                       "k2": 0.0022,

                       "rho_J": 1.2,
                       "rho_A": params_goehring["rho_A"],
                       "rho_P": params_goehring["rho_P"],

                       "sigmaJ": 1, "sigmaM": 1, "sigmaP": 1,
                       "alpha": 1, "beta": 2,
                       }

# parameters we will vary
PARAMS_TO_VARY = ["kJP", "kMP", "kAP", "konJ",
                  "koffJ", "koffM", "koffA",
                  "k1", "k2",
                  "rho_J"
                  ]

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

OUTPUT_FOLDER = "output_parameter_search"
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

        best_point = do_variations(params_par3add, goehring_res, variation_pairs, variation_multipliers)

        print(f"Finished run {rep_i+1}/{MAX_REPETITIONS}")
        print(f"Best point was {best_point}")

        params_par3add = {**params_par3add, **best_point[0]}

        if best_point is None:
            break

    plt.show()


def do_variations(params_par3add, goehring_results: tuple[list], variation_pairs: list[tuple],
                  variation_multipliers: tuple[dict, float]) -> dict:

    best_point: tuple[dict, float] = None

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
                                     "variation_info": {p1: ppair[0], p2: ppair[1]}
                                     })
            tasks_hom.append(task)

        # load or run to get results
        res_hom_all = load_or_run(f"{LABEL_P_HOM}_{p1}_{p2}", tasks_hom)

        # generate polarisation tasks
        tasks_pol = []

        for res in res_hom_all:
            if res[1] != "FAILURE":
                varied_params_value = res[2]["variation_info"]
                init_cond_pol = res[1].y[:, -1]

                # takes long time, so skipping stuff that's way off
                hsim = calculate_similarity_gp(goehring_results[0], init_cond_pol)
                if hsim > 5:
                    continue

                task = (MODELS.PAR3ADD, {**params_par3add, **varied_params_value,
                                         "variation_info": varied_params_value,
                                         "Nx": NX, "tL": TL_EST, "initial_condition": init_cond_pol,
                                         "label": f"{LABEL_P_POL}_{p1}_{varied_params_value[p1]:.4f}_{p2}_{varied_params_value[p2]:.4f}",
                                         })
                tasks_pol.append(task)
            else:
                print("failure detected")

        # load or run to get results
        res_pol_all = load_or_run(f"{LABEL_P_POL}_{p1}_{p2}", tasks_pol)

        # compare with goehring
        comparisons = goehring_comparer(goehring_results, res_hom_all, res_pol_all)

        for c in comparisons:
            if best_point is None or best_point[1] > c[2]:  # found lower polarised difference
                best_point = (c[0], c[2])

        # plot
        if not NO_PLOT:
            print("\n".join([f"{c}" for c in comparisons]))

            # plot for homogeneous comparison
            plot_gcomparisons(p1, p2, [(c[0], c[1]) for c in comparisons], (params_par3add[p1], params_par3add[p2]))
            # save plot
            fig_file_name = f"./{OUTPUT_FOLDER}/{f'{time.time_ns()}'[5:]}{LABEL_P_HOM}_compare_{p1}_{p2}"
            plt.savefig(fig_file_name)
            print(f"Saved figure to {fig_file_name}")
            plt.close("all")

            # plot for polarised comparison
            plot_gcomparisons(p1, p2, [(c[0], c[2]) for c in comparisons], (params_par3add[p1], params_par3add[p2]))
            # save plot
            fig_file_name = f"./{OUTPUT_FOLDER}/{f'{time.time_ns()}'[5:]}{LABEL_P_POL}_compare_{p1}_{p2}"
            plt.savefig(fig_file_name)
            print(f"Saved figure to {fig_file_name}")
            plt.close("all")

    # animate plot for best detected parameter pair
    if True and best_point is not None:
        for res in res_pol_all:
            if res[1] != "FAILURE" and res[2]["variation_info"] == best_point[0]:
                print("Animating plot of best point")
                MODULE_PAR3ADD.animate_plot(res[1], res[2],
                                            save_file=True,
                                            file_code=f"./{OUTPUT_FOLDER}/best_point_{best_point[0]}")
                break

    return best_point


def plot_gcomparisons(p1, p2, comparisons: list[tuple[dict, float]], baseline_point):
    plt.figure()

    max_val = np.max([[v[1]] for v in comparisons])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["violet", "blue", "yellow"])
    norm = plt.Normalize(0, max_val)

    for point in comparisons:
        v1 = point[0][p1]
        v2 = point[0][p2]
        comp_val = point[1]
        plt.scatter(v1, v2, c=comp_val, cmap=cmap, norm=norm, marker="o", s=100)

    plt.scatter(baseline_point[0], baseline_point[1], c=max_val, cmap=cmap,
                norm=norm, marker=".", s=100)  # plot existing params point for comparison

    clb = plt.colorbar()
    clb.ax.set_title(f"g_diff max:{max_val:.2f}")

    plt.xlabel(p1)
    plt.ylabel(p2)
    plt.title(f"2sensitivity vs goehring varied {p1} and {p2}")


def goehring_comparer(goehring_results: tuple[list], res_hom_all: list[tuple],
                      res_pol_all: list[tuple]) \
                        -> list[tuple[dict, float, float]]:
    assert len(res_pol_all) <= len(res_hom_all)

    out_list: list[tuple[dict, float, float]] = []

    for res in res_pol_all:
        if res[1] != "FAILURE":
            varied_param_values: dict = res[2]["variation_info"]

            goehring_sim_hom = None

            # find equivalent result from homogeneous run
            for hom_res in res_hom_all:
                if hom_res[1] != "FAILURE":
                    hom_variation_info = hom_res[2]["variation_info"]
                    if hom_variation_info == varied_param_values:
                        goehring_sim_hom = calculate_similarity_gp(goehring_results[0], hom_res[1].y[:, -1])

            # calculate goehring simularity (polarised)
            goehring_sim_pol = calculate_similarity_gp(goehring_results[1], res[1].y[:, -1])

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
                goehring_sim_partway = calculate_similarity_gp(full_g_res_tuple[0], res[1].y[:, int(res[1].y.shape[1]//full_g_res_tuple[1])])
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


def get_goehring_res(pol_time_divider: int = 1) -> tuple[list]:
    try:
        loaded = np.load("./savedata/goehring_for_par3add_2_sensitivity200225.npy", allow_pickle=True)
        print("Loaded goehring results")
        return loaded
    except Exception as e:
        print("Failed to load goehring: " + str(e))
        res_h = get_goehring_homo_ic()
        res_p = get_goehring_polarised(res_h, pol_time_divider)
        np.save("./savedata/goehring_for_par3add_2_sensitivity200225.npy", (res_h, res_p), allow_pickle=True)
        return (res_h, res_p)


def get_goehring_homo_ic():
    task = (MODELS.GOEHRING, {**params_goehring, "tL": TL_HOM,
                              "initial_condition": [1]*NX + [0]*NX, "v_func": v_func_zero})
    res = model_task_handler.run_tasks([task])[0]
    return res[1].y[:, -1]


def get_goehring_polarised(initial_condition, pol_time_divider: int = 1):
    task = (MODELS.GOEHRING, {**params_goehring, "tL": TL_EST,
                              "initial_condition": initial_condition})
    res = model_task_handler.run_tasks([task])[0]
    # return res[1].y[:, -1] if not pol_half_time else res[1].y[:, res[1].y.shape(1)//pol_time_divider]
    return res[1].y[:, res[1].y.shape(1)//pol_time_divider]


def load_or_run(name: str, tasks: list[tuple]) -> list[tuple]:
    filename = f"{name}_{bad_hash_for_filename(tasks):.7g}"

    try:
        loaded_data = np.load("./savedata/"+filename+".npy", allow_pickle=True)
        print(f"Loading of {name} succeeded!")
        return loaded_data
    except Exception as e:
        print(f"Failed loading of {name} because: " + str(e))
        res = model_task_handler.run_tasks_parallel(tasks)

        print("Saving results")
        np.save("./savedata/"+filename+".npy", res, allow_pickle=True)

        return res


def bad_hash_for_filename(tasks):
    bad_hash = 0

    for task in tasks:
        params = task[1]
        for key in params:
            element = params[key]
            if isinstance(element, (int, float)):
                bad_hash += element
            elif key == "initial_condition":
                bad_hash += np.sum(element)
    bad_hash += len(tasks)
    return bad_hash


if __name__ == '__main__':
    matplotlib.use('Agg')  # block plots from appearing
    main()

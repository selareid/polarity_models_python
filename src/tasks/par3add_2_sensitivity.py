# Want to find good parameters for the new model
# it is close to goehring, but too far to the right
import os
import time
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from numpy import linalg
from ..models import MODELS, model_to_module, model_to_string
import matplotlib
from matplotlib import pyplot as plt
from src import model_task_handler


MODULE_PAR3ADD = model_to_module(MODELS.PAR3ADD)

params_goehring = { "psi": 0.174,
                    "D_A": 0.28, "D_P": 0.15,
                    "k_onA": 8.58 * 10 ** (-3), "k_onP": 4.74 * 10 ** (-2),
                    "k_offA": 5.4 * 10 ** (-3), "k_offP": 7.3 * 10 ** (-3),
                    "k_AP": 0.190, "k_PA": 2.0,
                    "rho_A": 1.56, "rho_P": 1.0,
                    "alpha": 1, "beta": 2,
                    }
params_par3add = {"psi": params_goehring["psi"],
                  "D_J": params_goehring["D_A"],
                  "D_M": params_goehring["D_P"]/2, # this should be less diffusive, 0 is bad tho
                  "D_A": params_goehring["D_A"],
                  "D_P": params_goehring["D_P"],

                  "kJP": params_goehring["k_AP"]/2,
                  "kMP": params_goehring["k_AP"]/2,
                  "kAP": params_goehring["k_AP"]/2,
                  "kPA": params_goehring["k_PA"],

                  "konJ": params_goehring["k_onA"],
                  "konA": 0, # not used in writeup
                  "konP": params_goehring["k_onP"],

                  "koffJ": params_goehring["k_offA"]/2,
                  "koffM": params_goehring["k_offA"],
                  "koffA": params_goehring["k_offA"],
                  "koffP": params_goehring["k_offP"],

                  "k1": params_goehring["k_onA"],
                  "k2": params_goehring["k_onA"],

                  "rho_J": 1.2,
                  "rho_A": params_goehring["rho_A"],
                  "rho_P": params_goehring["rho_P"],

                  "sigmaJ": 1,"sigmaM": 1,"sigmaP": 1,
                  "alpha": 1, "beta": 2,
                  }

PARAMS_TO_VARY = ["kJP","kMP","kAP","konJ","koffJ","k1","k2","rho_J"]

MAX_MULTIPLIER = 1.5
MIN_MULTIPLIER = 0.5
TOTAL_STEPS = 13

NX = 100
TL_HOM = 3000
TL_EST = 9000

# IC for par3add
INIT_COND_HOM = [1]*NX + [1]*(2*NX) + [0]*NX

LABEL_P_HOM = "200225_par3add_hom_run"
LABEL_P_POL = "200225_par3add_pol_run"

OUTPUT_FOLDER = "210225"
NO_PLOT = True # just do sim

def v_func_zero(kvals, x, t):
    return 0

def main():
    matplotlib.use('Agg') # block plots from appearing

    variation_multipliers = [MIN_MULTIPLIER+i*(MAX_MULTIPLIER-MIN_MULTIPLIER)/(TOTAL_STEPS-1) for i in range(TOTAL_STEPS)]
    # print(variation_multipliers)

    variation_pairs = []

    for i in range(len(PARAMS_TO_VARY)):
        for j in range(i+1, len(PARAMS_TO_VARY)):
            variation_pairs.append((PARAMS_TO_VARY[i], PARAMS_TO_VARY[j]))

    print(variation_pairs)

    print("Getting goehring results")
    goehring_res = get_goehring_res()

    print("Beginning variations")
    do_variations(goehring_res, variation_pairs, variation_multipliers)

    # plt.show(block=True)


def do_variations(goehring_results: tuple[list], variation_pairs: list[tuple], variation_multipliers: list):
    for i_variation_pair in range(len(variation_pairs)):
        variation_pair = variation_pairs[i_variation_pair]

        # == Generate variation sets ==
        p1 = variation_pair[0]
        p2 = variation_pair[1]

        varied_params = {p1: params_par3add[p1], p2: params_par3add[p2]}

        print(f"Working on {varied_params} {i_variation_pair+1}/{len(variation_pairs)}")

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
                init_cond_pol = res[1].y[:,-1]
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

        if not NO_PLOT:
            # compare with goehring
            comparisons = goehring_comparer(goehring_results, res_hom_all, res_pol_all)

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


def plot_gcomparisons(p1, p2, comparisons: list[tuple[dict,float]], baseline_point):
    plt.figure()

    max_val = np.max([[v[1]] for v in comparisons])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["violet", "blue"])
    norm = plt.Normalize(0, max_val)

    for point in comparisons:
        v1 = point[0][p1]
        v2 = point[0][p2]
        comp_val = point[1]
        plt.scatter(v1, v2, c=comp_val, cmap=cmap, norm=norm, marker="o", s=100)
    
    plt.scatter(baseline_point[0], baseline_point[1], c="black", marker=".", s=100) # plot existing params point for comparison

    clb = plt.colorbar()
    clb.ax.set_title("g_diff")

    plt.xlabel(p1)
    plt.ylabel(p2)
    plt.title(f"2sensitivity vs goehring varied {p1} and {p2}")

def goehring_comparer(goehring_results: tuple[list], res_hom_all: list[tuple], res_pol_all: list[tuple]) -> list[tuple[dict, float, float]]:
    assert len(res_pol_all) <= len(res_hom_all)

    out_list = []

    for res in res_pol_all:
        if res[1] != "FAILURE":
            varied_param_values: dict = res[2]["variation_info"]
            
            goehring_sim_hom = None

            # find equivalent result from homogeneous run
            for hom_res in res_hom_all:
                if hom_res[1] != "FAILURE":
                    hom_variation_info = hom_res[2]["variation_info"]
                    if hom_variation_info == varied_param_values:
                        goehring_sim_hom = calculate_similarity_gp(goehring_results[0], hom_res[1].y[:,-1])
            
            # calculate goehring simularity (polarised)
            goehring_sim_pol = calculate_similarity_gp(goehring_results[1], res[1].y[:,-1])

            if goehring_sim_hom != None:
                out_list.append((varied_param_values, goehring_sim_hom, goehring_sim_pol))
        else:
            print("failure detected")

    return out_list

def calculate_similarity_gp(goehring_res: list, par3add_res: list):
    return linalg.vector_norm(goehring_res - 
                              np.concatenate((par3add_res[NX:2*NX] + par3add_res[2*NX:3*NX], par3add_res[3*NX:])))


def get_goehring_res() -> tuple[list]:
    try:
        loaded = np.load("./savedata/goehring_for_par3add_2_sensitivity200225.npy", allow_pickle=True)
        print("Loaded goehring results")
        return loaded
    except Exception as e:
        print("Failed to load goehring: " + str(e))
        res_h = get_goehring_homo_ic()
        res_p = get_goehring_polarised(res_h)
        np.save("./savedata/goehring_for_par3add_2_sensitivity200225.npy", (res_h,res_p), allow_pickle=True)
        return (res_h, res_p)

def get_goehring_homo_ic():
    task = (MODELS.GOEHRING, {**params_goehring, "tL": TL_HOM,
                                "initial_condition": [1]*NX + [0]*NX, "v_func": v_func_zero,})
    res = model_task_handler.run_tasks([task])[0]
    return res[1].y[:,-1]

def get_goehring_polarised(initial_condition):
    task = (MODELS.GOEHRING, {**params_goehring, "tL": TL_EST,
                                "initial_condition": initial_condition})
    res = model_task_handler.run_tasks([task])[0]
    return res[1].y[:,-1]


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
    main()
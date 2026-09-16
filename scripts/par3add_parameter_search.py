# Discover parameters by iterative grid search

import time
import numpy as np
from numpy import linalg
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
from pickle import dump, load
from typing import NamedTuple
from scipy.optimize import OptimizeResult
from dataclasses import asdict

from polarity.utilities import model_task_handler, figure_helper as fh
from polarity.model_enums import MODELS, model_to_module

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
                       "kdisM": 0.0022,

                       "rho_J": 1.2,
                       "rho_A": params_goehring.rho_A,
                       "rho_P": params_goehring.rho_P,

                       "sigmaJ": 1, "sigmaM": 1, "sigmaP": 1,
                       "alpha": 1, "beta": 2,
                       }

# parameters we will vary
PARAMS_TO_VARY = ["kJP", "kMP", "kAP", "konJ",
                  "koffJ", "koffM", "koffA",
                  "konM", "kdisM",
                  "rho_J"
                  ]

# set range of parameter variation and number of repetitions
MULTIPLIERS = np.linspace(0.75, 1.25, 4).tolist()
MAX_REPETITIONS = 25  # max iterations of full variations run-through

# Simulation durations and grid size
TL_HOM = 3000   # end time (in seconds) for homogeneous simulation
TL_EST = 120*60 # end time (in seconds) for establishment simulation
NX = 101        # number of X voxels

# Time points used for comparisons and evaluation times
COMPARISON_TIME_POINTS = [ mins*60 for mins in [10, 20, 40, 60, 80, 100] ]
T_EVAL = sorted(set(COMPARISON_TIME_POINTS + [0, TL_EST])) 

# Labels and folders for saving results
LABEL_P_HOM = "par3add_hom_run"
LABEL_P_POL = "par3add_pol_run"

FIG_OUTPUT_FOLDER = Path( fh.FIGURES_DIR / "par3add_parameter_search" )
DAT_OUTPUT_FOLDER = Path( fh.DATA_DIR / "par3add_parameter_search_updated")

FORCE_RERUN = False  # if True, will re-run all tasks even if results are already saved
SAVE_SIMILARITY_DATA = True  # if True, will save all the iterations and similarity results iteratively into a pickle
NO_PLOT = True 
N_PROC = 16 # Number of processes to use for parallel runs

class ModelResult(NamedTuple):
    hom_ic: list[float] # The homogeneous profile (used as the initial condition)
    sol: OptimizeResult # The PDE solution of the establishment simulation

class PointResult(NamedTuple):
    params: dict[str, float] # The parameter values for this point
    similarity: float # The similarity score for this point


def v_func_zero(kvals, x, t):
    return 0


def main():

    # Create output folders if they don't exist
    if not FIG_OUTPUT_FOLDER.exists():
        FIG_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    if not DAT_OUTPUT_FOLDER.exists():
        DAT_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Initialise the pickle file as empty to save the results of the parameter search
    if SAVE_SIMILARITY_DATA:
        with open(DAT_OUTPUT_FOLDER / "similarity_data.pkl", 'wb') as f:
            dump([], f)

    # Generate the parameter pairs to vary
    parameter_pairs = [(param_i, param_j) for i, param_i in enumerate(PARAMS_TO_VARY) 
                                            for j, param_j in enumerate(PARAMS_TO_VARY) if j > i ]

    # Get Goehring simulation data for comparison
    print("Getting Goehring results")
    goehring_hom_ic = MODULE_GOEHRING.run_for_ss_initial_condition({"Nx": NX}, tL=TL_HOM)
    goehring_sol, _ = MODULE_GOEHRING.run_model( {"Nx": NX, "tL": TL_EST, "t_eval": T_EVAL,
                                                  "initial_condition": goehring_hom_ic} )
    goehring_result = ModelResult(hom_ic=goehring_hom_ic, sol=goehring_sol)
    with open(DAT_OUTPUT_FOLDER / f"goehring_results.pkl", 'wb') as f:
        dump(goehring_result, f)


    # Run parameter variations
    print("Beginning variations")
    params_par3add = base_params_par3add
    with open(DAT_OUTPUT_FOLDER / "best_point.pkl", 'wb') as f:
            dump((0, None, params_par3add), f)
    best_point = None

    for rep_i in range(MAX_REPETITIONS):  # Iterate multiple consecutive runs
        print(f"Running run number {rep_i+1}/{MAX_REPETITIONS}")
        best_point_changed = False # Flag to track if the best point has changed in this iteration

        # Iterate over the parameter pairs and get best point for each pair
        
        for i_pair, ppair in enumerate(parameter_pairs):
            print(f"{i_pair+1}/{len(parameter_pairs)} - Working on {ppair}")
            bp_ppair = do_variations(ppair, params_par3add, goehring_result, rep_i)
            if best_point is None or best_point.similarity > bp_ppair.similarity:  # found lower polarised difference
                best_point = bp_ppair
                best_point_changed = True
        print(f"Finished run {rep_i+1}/{MAX_REPETITIONS}. \n Best point was {best_point.params} with similarity {best_point.similarity:.4f}.")

        if best_point is None:
            raise Exception("No best point found. Something went wrong.")

        params_par3add = {**params_par3add, **best_point.params}

        # Add best point to file
        with open(DAT_OUTPUT_FOLDER / "best_point.pkl", 'ab') as f:
            dump((rep_i+1, best_point, params_par3add), f)

        # Break the loop if the best point hasn't changed in this iteration
        if not best_point_changed:
            print(f"No improvement in best point found in iteration {rep_i+1}. Stopping further iterations.")
            break

    # Print final point to console
    print("\033[1;32mFinal parameter set:\033[0m")
    for key, value in params_par3add.items():
        print(f"{key}: {value:.4f}")

# End main


def do_variations(param_pair: tuple[str, str], params_par3add, goehering_results: ModelResult, iter_id: int) -> PointResult: 

    p1, p2 = param_pair
    varied_params = {p1: params_par3add[p1], p2: params_par3add[p2]}
    best_point = None
    similarity_data = []

    # Create all parameter combos
    param_pairs = [{p1: var1*varied_params[p1], p2: var2*varied_params[p2]} 
                    for var1 in MULTIPLIERS for var2 in MULTIPLIERS]
    print(len(param_pairs), "parameter pairs to run for", p1, p2)

    # Run for homogeneous initial condition first
    tasks_hom = [ ( MODELS.PAR3ADD, {**params_par3add, **ppair_dict,
                                        "Nx": NX, "tL": TL_HOM, "t_eval": [0, TL_HOM], 
                                        "v_func": v_func_zero,
                                        "label": f"{LABEL_P_HOM}_{p1}_{ppair_dict[p1]:.4f}_{p2}_{ppair_dict[p2]:.4f}"
                                        } ) 
                                        for ppair_dict in param_pairs ]
    res_hom_all = load_or_run(f"{LABEL_P_HOM}_{p1}_{p2}_{iter_id + 1}", tasks_hom)

    # Generate the establishment simulation task list
    tasks_pol = []
    for label, (sol, setup) in res_hom_all:
        if sol is None:
            raise Exception(f"Simulation failed for label {label}. Check logs for details.")
        
        varied_params_value = {p1: setup[p1], p2: setup[p2]}
        init_cond_pol = sol.y[:, -1]

        # Check for similarity and skip any that are way off
        hsim = calculate_similarity_gp(goehering_results.hom_ic, init_cond_pol)
        if hsim > 5:
            continue

        # If it is a potential solution add to the task list
        task = (MODELS.PAR3ADD, {**params_par3add, **varied_params_value,
                                    "Nx": NX, "tL": TL_EST, "t_eval": T_EVAL,
                                    "initial_condition": init_cond_pol,
                                    "label": f"{LABEL_P_POL}_{p1}_{varied_params_value[p1]:.4f}_{p2}_{varied_params_value[p2]:.4f}"
                                    })
        tasks_pol.append(task)

    # Load or run to get results
    res_pol_all = load_or_run(f"{LABEL_P_POL}_{p1}_{p2}_{iter_id+1}", tasks_pol)

    # Compare the polarised solutions with the Goehring results and update best point
    for label, (pol_sol, setup) in res_pol_all:
        if pol_sol is None:
            raise Exception(f"Simulation failed for label {label}. Check logs for details.")
        candidate_result = ModelResult(hom_ic=setup["initial_condition"], sol=pol_sol)
        sim_hom, sim_pol = goehring_comparer(goehering_results, candidate_result )
        if best_point is None or best_point.similarity > sim_pol:  # found lower polarised difference
            best_point = PointResult(params = {p1: setup[p1], p2: setup[p2]}, similarity = sim_pol )
        similarity_data.append( ( {p1: setup[p1], p2: setup[p2]}, sim_hom, sim_pol ) )

    # Print the best point for this parameter pair
    if best_point is None:
        raise Exception(f"No best point found for parameter pair {p1}, {p2}. Something went wrong.")
    print(f"Best point for {p1}, {p2} was {best_point.params} (similarity: {best_point.similarity:.4f}).")

    # Save the similarity data for this parameter pair
    if SAVE_SIMILARITY_DATA:
        with open(DAT_OUTPUT_FOLDER / f"similarity_data.pkl", 'ab') as f:
            dump( { "iteration": iter_id, "parameter_pair": (p1, p2), 
                   "similarity_data": similarity_data, "best_point": best_point }, f)

    # Save the plots / animations if desired
    if not NO_PLOT:
        plot_similarity(p1, p2, similarity_data, (params_par3add[p1], params_par3add[p2]))
        animated_plot(params_par3add, best_point)

    return best_point



def goehring_comparer(goehring_results: ModelResult, comparison_results: ModelResult) \
                        -> tuple[float, float]:

    # Calculate the similarity for the homogeneous results
    similarity_hom = calculate_similarity_gp(goehring_results.hom_ic, comparison_results.hom_ic)

    # Calculate the similarity for the final polarised profile
    assert goehring_results.sol.t[-1] == comparison_results.sol.t[-1], "Final time points do not match for Goehring and comparison results."
    similarity_pol_final = calculate_similarity_gp(goehring_results.sol.y[:,-1], comparison_results.sol.y[:, -1])

    # Calculate the similarity for the midway time points
    similarity_midway_points = np.empty(len(COMPARISON_TIME_POINTS))
    i0 = 0 if COMPARISON_TIME_POINTS[0] == T_EVAL[0] else 1 # First time point in t_eval is probably 0
    for i, comp_time in enumerate(COMPARISON_TIME_POINTS):
        assert goehring_results.sol.t[i0 + i] == comp_time, f"Goehring time points do not match: {goehring_results.sol.t[i0 + i]} != {comp_time}"
        assert comparison_results.sol.t[i0 + i] == comp_time, f"Comparison time points do not match: {comparison_results.sol.t[i0 + i]} != {comp_time}"
        similarity = calculate_similarity_gp(goehring_results.sol.y[:, i0 + i], comparison_results.sol.y[:, i0 + i])
        similarity_midway_points[i] = similarity
    similarity_pol_midway = np.mean(similarity_midway_points)

    # The similarity for the polarised profile is the average of the final polarised profile and the midway time points
    similarity_pol = (similarity_pol_final + similarity_pol_midway) / 2

    return (similarity_hom, similarity_pol)



def calculate_similarity_gp(goehring_profile: list, par3add_profile: list):
    par3add_Astar = np.add( par3add_profile[NX:2*NX], par3add_profile[2*NX:3*NX] ) # M + A
    par3add_Astar_P = np.concatenate((par3add_Astar, par3add_profile[3*NX:]))
    assert len(goehring_profile) == len(par3add_Astar_P), "Goehring and Par3add [A*, P] have mismatched lengths."
    return linalg.vector_norm(goehring_profile - par3add_Astar_P)



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
        res = model_task_handler.run_tasks_parallel(tasks, NUMBER_OF_PROCESSES=N_PROC)
        for label, (pol_sol, setup) in res:
            if pol_sol is None:
                raise Exception(f"Simulation failed for label {label}. Check logs for details.")
        print(f"Saving results {name}")
        with open(filename, 'wb') as f:
            dump(res, f)
        return res



def plot_similarity(p1, p2, similarity_data: list[tuple[dict, float, float]], baseline_point):

    # Extract similarity data
    p1_dat = [ppair[p1] for ppair, hom, pol in similarity_data]
    p2_dat = [ppair[p2] for ppair, hom, pol in similarity_data]
    similarity_data_hom = [hom for ppair, hom, pol in similarity_data]
    similarity_data_pol = [pol for ppair, hom, pol in similarity_data]

    # Now generate the plots for both homogeneous and polarised comparisons
    for label, data in zip ( [LABEL_P_HOM, LABEL_P_POL], [similarity_data_hom, similarity_data_pol] ):
        fig, ax = plt.subplots(figsize=(6, 4))
        
        max_val = np.max(data)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["violet", "blue", "yellow"])
        norm = plt.Normalize(0, max_val)

        ax.scatter(p1_dat, p2_dat, c=data, cmap=cmap, norm=norm, marker="o", s=100)

        sc = ax.scatter(baseline_point[0], baseline_point[1], c=max_val, cmap=cmap,
                    norm=norm, marker=".", s=100)  # plot existing params point for comparison

        clb = fig.colorbar(sc, ax=ax)
        clb.ax.set_title(f"g_diff max:{max_val:.2f}")

        ax.set_xlabel(p1)
        ax.set_ylabel(p2)
        ax.set_title(f"{p1}, {p2}")

        fig_file_name = FIG_OUTPUT_FOLDER/ f"{f'{time.time_ns()}'[5:]}{label}_compare_{p1}_{p2}"
        fig.savefig(fig_file_name)
        print(f"Saved figure to {fig_file_name}")



def animated_plot(base_params, best_point):
    # Re-solve with the higher density output
    sol, setup = MODULE_PAR3ADD.run_model({**base_params, **best_point.params, "Nx": NX, "tL": TL_EST}, calc_ss_initial_condition=True)
    # Animated plot
    print("Animating plot of best point")
    filename = "best_point-" + "-".join(f"{key}_{value:.5f}" for key, value in best_point.params.items())
    print(filename)
    fh.animate_plot(sol, asdict(setup), save_file = FIG_OUTPUT_FOLDER / filename )



if __name__ == '__main__':
    matplotlib.use('Agg')  # block plots from appearing
    main()

# See how tostevin changes depending on initial condition
from matplotlib import pyplot as plt
from .. import model_task_handler
from ..models import MODELS, model_to_module


def a_func_zero(kvals, lt, x): return 0


L = 50 # the default L, used for initial l(t=0) value
Nx = 150

# format of initial condition is [Am... Ac... Pm... Pc... L] with Nx of each A/P quantity
INITIAL_CONDITIONS = {
    "homogeneous_both":                 [1]*Nx                       + [1]*Nx                        + [1]*Nx                       + [1]*Nx                       + [L],
    "homogeneous_just_membrane":        [1]*Nx                       + [0]*Nx                        + [1]*Nx                       + [0]*Nx                       + [L],
    "homogeneous_just_cyto":            [0]*Nx                       + [1]*Nx                        + [0]*Nx                       + [1]*Nx                       + [L],

    # Below includes some adjustment in the non-peak term so that there are equal amounts of A/P overall
    "centre_membrane_peak_A":           [0]*(Nx//3)+[1]*(Nx-2*(Nx//3))+[0]*(Nx//3)  + [1]*Nx         + [(Nx-2*(Nx//3))/Nx]*Nx                      + [1]*Nx        + [L],
    "centre_membrane_peak_P":           [(Nx-2*(Nx//3))/Nx]*Nx                      + [1]*Nx         + [0]*(Nx//3)+[1]*(Nx-2*(Nx//3))+[0]*(Nx//3)  + [1]*Nx        + [L],

    "polarised_even_only_membrane":     [1]*(Nx//2)+[0]*(Nx - Nx//2) + [0]*Nx                        + [0]*(Nx//2)+[1]*(Nx - Nx//2) + [0]*Nx                       + [L],
    "polarised_even_only_cyto":         [0]*Nx                       + [1]*(Nx//2)+[0]*(Nx - Nx//2)  + [0]*Nx                       + [0]*(Nx//2)+[1]*(Nx - Nx//2) + [L],
    "polarised_even_both":              [1]*(Nx//2)+[0]*(Nx - Nx//2) + [1]*(Nx//2)+[0]*(Nx - Nx//2)  + [0]*(Nx//2)+[1]*(Nx - Nx//2) + [0]*(Nx//2)+[1]*(Nx - Nx//2) + [L],

    # put A on right, P on left
    "polarised_even_both_flipped":      [0]*(Nx//2)+[1]*(Nx - Nx//2) + [0]*(Nx//2)+[1]*(Nx - Nx//2)  + [1]*(Nx//2)+[0]*(Nx - Nx//2) + [1]*(Nx//2)+[0]*(Nx - Nx//2) + [L],

    "just_A_homogeneous":               [1]*Nx                       + [1]*Nx                        + [0]*Nx                       + [0]*Nx                       + [L],
    "just_A_left_skew":                 [1]*(Nx//2)+[0]*(Nx - Nx//2) + [1]*(Nx//2)+[0]*(Nx - Nx//2)  + [0]*Nx                       + [0]*Nx                       + [L],
    "just_A_right_skew":                [0]*(Nx//2)+[1]*(Nx - Nx//2) + [0]*(Nx//2)+[1]*(Nx - Nx//2)  + [0]*Nx                       + [0]*Nx                       + [L],

    "just_P_homogeneous":               [0]*Nx                       + [0]*Nx                        + [1]*Nx                       + [1]*Nx                       + [L],
    "just_P_left_skew":                 [0]*Nx                       + [0]*Nx                        + [1]*(Nx//2)+[0]*(Nx - Nx//2) + [1]*(Nx//2)+[0]*(Nx - Nx//2) + [L],
    "just_P_right_skew":                [0]*Nx                       + [0]*Nx                        + [0]*(Nx//2)+[1]*(Nx - Nx//2) + [0]*(Nx//2)+[1]*(Nx - Nx//2) + [L],
    }

END_TIME = 500

TASKS_DEFAULT_A = [(MODELS.TOSTEVIN, {"points_per_second": 1500/END_TIME, "tL": END_TIME, "Nx": Nx, "L": L, "label": "tostevin_"+key,
    "initial_condition": INITIAL_CONDITIONS[key]}) for key in INITIAL_CONDITIONS]
TASKS_ZERO_A = [(MODELS.TOSTEVIN, {"points_per_second": 1500/END_TIME, "tL": END_TIME, "Nx": Nx, "L": L, "label": "tostevin_"+key+"_a=0",
    "a_func": a_func_zero, "initial_condition": INITIAL_CONDITIONS[key]}) for key in INITIAL_CONDITIONS]

if __name__ == '__main__':
    # results = model_task_handler.run_tasks_parallel(TASKS_DEFAULT_A)
    results = model_task_handler.run_tasks_parallel(TASKS_ZERO_A)

    results.sort(key=lambda res: 0 if res[1]=="FAILURE" or not "sort" in res[2] else res[2]["sort"])  # order when plotting multiple solutions on single figure

    sol_list = []
    kvals_list = []

    for res in results:
        if res[1] == "FAILURE":
            continue

        # model_module = model_to_module(res[0])

        ### Plots and Animated Plots for individual solutions
        # model_module.plot_overall_quantities_over_time(res[1], res[2])
        # model_module.plot_final_timestep(res[1], res[2])
        # model_module.animate_plot(res[1], res[2], save_file=True, as_proportion=True, file_code=res[2]["label"])
        # model_module.plot_lt(res[1], res[2])

        sol_list.append(res[1])
        kvals_list.append(res[2])

    # Output Final Timestep for All Solutions Together
    model_module = model_to_module(MODELS.TOSTEVIN)
    model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Ac=False,plot_Pm=False,plot_Pc=False, label="tostevin_zero_a differing IC")
    # model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Am=False,plot_Pm=False,plot_Pc=False)
    # model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Am=False,plot_Ac=False,plot_Pc=False)
    # model_module.plot_multi_final_timestep(sol_list, kvals_list, plot_Am=False,plot_Ac=False,plot_Pm=False)
    # model_module.plot_multi_final_timestep(sol_list, kvals_list)


    plt.show()  # if we don't have this the program ends and the plots close
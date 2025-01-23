import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from src import model_task_handler
from matplotlib import pyplot as plt
from ..models import MODELS, model_to_module
from src.tasks import variation_task_helper as t_helper

Module_Par3Add = model_to_module(MODELS.PAR3ADD)

def v_func_zero(kvals, x, t):
    return 0

base_params = {
    "psi": 0.174,
    "D_J": 0.28,"D_M": 0.28,"D_A": 0.28,"D_P": 0.15,
    "sigmaJ": 1,"sigmaM": 1,"sigmaP": 1,
    "k1": 8.58*10**(-3), "k2": 7.5*10**(-3),
    "kAP": 0.19, "kJP": 0.0010, "kMP": 0.006, "kPA": 0.1196,
    "alpha": 1, "beta": 2,
    "rho_A": 1.56, "rho_J": 0.7, "rho_P": 1,
    "konA": 0, "konJ": 0.00858, "konP": 0.0474,
    "koffA": 0.00545, "koffJ": 0.001, "koffP": 0.008,
}


def test_polarity_establishment(filename, Nx, tL, initial_condition, force_run = False):
    variation_multipliers, index_for_100x = t_helper.get_variation_multiplier()
    plot_xticks = t_helper.get_xticks(variation_multipliers, index_for_100x)

    tasks = t_helper.generate_tasks(MODELS.PAR3ADD, variation_multipliers, base_params,
                                    {
                                        "initial_condition": initial_condition,
                                        # "label": "par3add",
                                        "Nx": Nx,
                                        "tL": tL,
                                        # "v_func": ,
                                    })

    print("Testing polarity establishment")
    print(f"{len(tasks)} sets of tasks to run totalling {sum([len(tasks[i]) for i in range(0,len(tasks))])} tasks")

    print("Attempting load from save")
    load_data = t_helper.load_runs(filename)
    if not force_run and load_data[0]:
        print("Loading succeeded")
        baseline = load_data[1]
        results_by_variable = load_data[2]
    else:
        print("Loading failed")
        baseline, results_by_variable = t_helper.split_baseline_from_results(MODELS.PAR3ADD,
                                                        t_helper.run_grouped_tasks(tasks), index_for_100x)
        
        print("Saving run data")
        t_helper.save_runs(filename, tasks, baseline, results_by_variable)
        print("Saving finished")
    

    print("Plotting Results")
    Module_Par3Add.plot_variation_sets(results_by_variable, 
                                    label="Par3Add Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"",
                                    x_axis_labels=plot_xticks)
    Module_Par3Add.plot_variation_sets(results_by_variable, 
                                    label="Par3Add Polarity Establishment Nx="+str(Nx)+",tL="+str(tL)+"",
                                    x_axis_labels=plot_xticks, xlim=["60%", "140%"])

    return baseline


if __name__ == '__main__':
    Nx = 100
    tL_establishment = 3000
    filename = "par3addition_sens_anal"

    # Get initial condition
    homogeneous_ic = [0]*Nx + [0]*Nx + [0]*Nx + [0]*Nx
    parameters = {
            **base_params,
            "points_per_second": 0.1,
            "Nx": Nx,
    }

    ## run without advection to get initial condition (IC)
    print("Getting initial condition")
    
    task_initial_condition = (MODELS.PAR3ADD, {
            **parameters, "initial_condition": homogeneous_ic, "label": "par3add finding IC", "tL": 5000, "v_func": v_func_zero
        })
    
    res_initial_condition = model_task_handler.run_tasks([task_initial_condition])[0]
    stable_initial_condition = res_initial_condition[1].y[:,-1]


    baseline_sim_res = test_polarity_establishment(filename, Nx, tL_establishment, stable_initial_condition)

    # plot baseline run stuff
    Module_Par3Add.animate_plot(baseline_sim_res[0], baseline_sim_res[1], save_file=True)
    Module_Par3Add.animate_plot_apar_combo(baseline_sim_res[0], baseline_sim_res[1], save_file=True)
    Module_Par3Add.plot_final_timestep_apar_combo(baseline_sim_res[0], baseline_sim_res[1])


    
    print('Finished')
    plt.show(block=False)
    input("Input to close")
    plt.close("all")


# create plot of final timesteps of runs from savedata of parameter variation tests
# it adds a grey ghost so we can confirm we're at steady-state
# ffmpeg can stitch these together to get video of end state as parameter value changes
# Can use command >ffmpeg -framerate 3/1 -i %d.png output.mp4

from ..models import MODELS, model_to_module
from src.tasks import variation_task_helper
import matplotlib
from matplotlib import pyplot as plt
from src import figure_helper


def v_func_zero(kvals, x, t):
    return 0


filename = "<savedata filename>"


def do_stuff(desired_key):
    load_data = variation_task_helper.load_runs(filename)

    assert load_data[0]

    baseline_data = load_data[1]
    baseline_desired_key_value = baseline_data[1][desired_key]

    results_by_variable = load_data[2]

    # Find line where desired key is varied
    for vary_group in results_by_variable:
        key_varied = vary_group[1][1]["key_varied"]

        if key_varied == desired_key:  # we found the key we want
            for i in range(0, len(vary_group[0])):
                sol = vary_group[0][i]
                kvals = vary_group[1][i]

                if kvals[desired_key] == baseline_desired_key_value:
                    kvals["label"] = f"{kvals["label"]} 'baseline' {desired_key}:{kvals[desired_key]:.4g}"

                if sol != "FAILURE":
                    # model_to_module(MODELS.PAR3ADD).plot_final_timestep(sol, kvals)
                    # model_to_module(MODELS.PAR3ADD).plot_final_timestep_w_traced_crosspoint(sol, kvals)
                    
                    plt.figure()
                    
                    Nx = kvals["Nx"]
                    X = kvals["X"]

                    for j in [0, 1]:
                        J = sol.y[:Nx, -j-1]
                        M = sol.y[Nx:2*Nx, -j-1]
                        A = sol.y[2*Nx:3*Nx, -j-1]
                        P = sol.y[3*Nx:, -j-1]

                        plt.plot(X, J, color=["grey", figure_helper.par3add_colours[0]][j])
                        plt.plot(X, M, color=["grey", figure_helper.par3add_colours[1]][j])
                        plt.plot(X, A, color=["grey", figure_helper.par3add_colours[2]][j])
                        plt.plot(X, P, color=["grey", figure_helper.par3add_colours[3]][j])
                    
                     
                    plt.savefig(f"./170325figs_maintenance/{desired_key}_{i}.png")
                    plt.close()
                
                if sol != "FAILURE":
                  model_to_module(MODELS.PAR3ADD).animate_plot(sol, kvals, save_file=True,
                                                               file_code=f"./170325figs_maintenance/{desired_key}_{i}_{kvals[desired_key]:.4g}")


def main():
    variation_params_par3add = [
        "psi",
        "D_J", "D_M", "D_A", "D_P", "sigmaJ", "sigmaM", "sigmaP",
        "k1", "k2", "kAP", "kJP", "kMP", "kPA", "alpha", "beta",
        "rho_A", "rho_J", "rho_P", "konA", "konJ", "konP", "koffM", "koffA", "koffJ", "koffP"
    ]

    for key in variation_params_par3add:
        do_stuff(key)
        print(f"{key}")


if __name__ == '__main__':
    matplotlib.use('Agg')  # block plots from appearing
    main()

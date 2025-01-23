import numpy as np
from src import model_task_handler
from src.models import MODELS, model_to_module, model_to_string


# standard variation multipliers
def get_variation_multiplier(option=None):
    # variation_multipliers = [0.1, 0.25, 0.5, 2, 5, 10] # wide
    # variation_multipliers = [0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3]  # narrow
    # if option == 1: # middle
    #     variation_multipliers = [0.5, 0.6, 0.7, 0.9, 1.2, 1.3, 1.5, 1.75, 2]
    #     index_for_100x = 4
    # else:  # option == 0 and others; default variation
    # variation_multipliers = [0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 1.025, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2, 5, 10]
    # index_for_100x = 9
    if option == "scribble":
        # variation_multipliers = [0.65, 0.8, 1.35, 1.7]
        variation_multipliers = (np.linspace(0.5, 1.80, 9).tolist()
                                 + np.linspace(0,0.5,4).tolist()
                                 + np.linspace(1.8,5,4).tolist()
                                 + np.linspace(5,10,7).tolist())
        # variation_multipliers = np.linspace(4, 5.8, 10).tolist()# + [0.01, 10]
        variation_multipliers.append(1)
    elif option == "sparse":
        variation_multipliers = np.arange(0.5, 1.5 + 0.5, 0.5).tolist() + [0, 0.01, 0.05, 0.1, 0.25] + [1.75, 2, 5, 10]
    else:
        variation_multipliers = np.arange(0.5, 1.5 + 0.05, 0.05).tolist() +[0, 0.01,0.05,0.1,0.25] + [1.75,2,5,10]

    variation_multipliers.sort()
    index_for_100x = np.searchsorted(variation_multipliers, 1, 'right')
    return variation_multipliers, index_for_100x


def get_xticks(variation_multipliers, index_for_100x):
    xticks = [format(f"{x * 100:.2f}").rstrip('0').rstrip('.')+"%" for x in variation_multipliers]
    xticks.insert(index_for_100x, "100%")
    return xticks


def generate_tasks(model_type: MODELS, variations_list: list[float], default_params: dict, other_param_vals: dict={}) -> list[list[tuple]]:
    sort_i = 1

    baseline_params = {**default_params, **other_param_vals}
    tasks: list[list(tuple)] = [[(model_type, {**baseline_params, "sort": sort_i, "variation_multiplier": 1})]]  # we add baseline task
    sort_i += 1

    for key in default_params.keys():
        task_list = []

        if key not in ["key_varied", "variation_multiplier", "label", "points_per_second", "Nx", "L", "x0", "xL", "t0", "tL", "v_func", "a_func"]:  # exclude run conditions
            for multiplier in variations_list:
                task_list.append((model_type, {**baseline_params,
                    "label": format(f"{key}={baseline_params[key] * multiplier:.4f}"),
                    "key_varied": format(f"{key}"), key: baseline_params[key] * multiplier,
                    "variation_multiplier": multiplier, "sort": sort_i}))

                sort_i += 1

            tasks.append(task_list)

    return tasks


# assume tasks have sort property "sort"
def run_grouped_tasks(tasks: list[list[tuple]]):
    tasks_collapsed = []

    for task_list_i in range(0,len(tasks)):
        for task in tasks[task_list_i]:
            task[1]["task_list_i"] = task_list_i
            tasks_collapsed.append(task)

    assert len(tasks_collapsed) == sum([len(task_list) for task_list in tasks])

    results = model_task_handler.run_tasks_parallel(tasks_collapsed)
    results.sort(key=lambda res: res[2]["sort"])

    results_by_variable = [[] for _ in range(0, len(tasks))]

    for res in results:
        task_list_i = res[2]["task_list_i"]
        results_by_variable[task_list_i].append(res)

    print(f"Runner output has length {len(results_by_variable)}")

    return results_by_variable


def split_baseline_from_results(model_type: MODELS, all_results_by_variable: list[list[tuple]], index_for_100x, extra_plot=None):
    if extra_plot is None:  # mutable default argument fix
        extra_plot = []

    baseline_result = None
    results_by_variable = []

    for result_set_i in range(0, len(all_results_by_variable)):
        print(f"Arranging results set {result_set_i}")

        all_results_by_variable[result_set_i].sort(key=lambda res: res[2]["sort"])

        is_baseline = ( result_set_i == 0 and len(all_results_by_variable[result_set_i]) == 1 )
        sol_list = []
        kvals_list = []
        has_failure = False

        for res in all_results_by_variable[result_set_i]:
            if res[1] == "FAILURE":
                has_failure = True
            #     continue
            elif res[2]["label"] in extra_plot:
                model_to_module(model_type).plot_final_timestep(res[1], res[2], rescale=False)
                model_to_module(model_type).animate_plot(res[1], res[2], rescale=False, save_file=True, file_code=format(f"{res[2]['label']},N={res[2]['Nx']},T={res[2]['tL']}"))


            if is_baseline:
                baseline_result = (res[1], res[2])
                # model_to_module(model_type).animate_plot(res[1], res[2], rescale=True)
                

            sol_list.append(res[1])
            kvals_list.append(res[2])

        assert len(sol_list) == len(kvals_list)

        if not is_baseline:
            assert baseline_result is not None
            # insert baseline into the results list
            sol_list.insert(index_for_100x, baseline_result[0])
            kvals_list.insert(index_for_100x, baseline_result[1])
            results_by_variable.append((sol_list, kvals_list))

    assert baseline_result is not None
    assert len(results_by_variable) == len(all_results_by_variable) - 1

    return baseline_result, results_by_variable


def save_runs(filename, tasks, baseline, results_by_variable):
    save_data = {
        "filename": filename,
        "tasks": tasks,
        "baseline": baseline,
        "results_by_variable": results_by_variable
        }

    np.save("./savedata/"+filename, save_data, allow_pickle=True)
    


def load_runs(filename):
    try:
        loaded_data = np.load("./savedata/"+filename+".npy", allow_pickle=True)
        loaded_data = loaded_data.item()

        return [True, loaded_data["baseline"], loaded_data["results_by_variable"]]
    except Exception as e:
        print("Error occurred while loading: " + str(e))
        return [False]


def generate_variation_save_filename(name: str, model: MODELS, Nx: int, end_time: int,
                                    point_per_second: float, params: dict, varied_params: list,
                                    initial_condition, tasks) -> str:

    bad_hash = 0

    for key in params:
        element = params[key]
        if isinstance(element, (int, float)):
            bad_hash += element

            if key in varied_params:
                bad_hash += 0.5*element
    
    bad_hash += initial_condition.sum()
    bad_hash += len(tasks)


    return ( name
            + "_" + model_to_string(model)
            + "_" + str(Nx)
            + "_" + str(end_time)
            + "_" + str(point_per_second)
            + "_" + f"{bad_hash:.7g}"
            )
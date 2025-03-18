import copy
import time
from multiprocessing import Process, Queue, cpu_count
from .models import model_to_module
import numpy as np


def worker(input, output):
    for model, args in iter(input.get, 'STOP'):
        print(f"{time.time():.1f} Running task for model: {model}")
        try:
            result = model_to_module(model).run_model(copy.deepcopy(args))
            output.put((model, result))
        except Exception as e:
            print(f"{time.time():.1f} Exception occurred while running task for model {model}; {e}")
            output.put((model, ("FAILURE",args)))


# Output of form [(model, sol, kvals),...]
def run_tasks_parallel(task_list, NUMBER_OF_PROCESSES=int(cpu_count()/1.5), callback=None) -> list[tuple]:
    assert NUMBER_OF_PROCESSES >= 1
    assert cpu_count() >= NUMBER_OF_PROCESSES

    # create queues
    task_queue = Queue()
    done_queue = Queue()

    # add tasks to queue
    for task in task_list:
        task_queue.put(task)

    print(f"{time.time():.1f} Running {len(task_list)} tasks on {NUMBER_OF_PROCESSES} processes")

    # start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    output_list = []

    # get and handle results (unordered)
    for i in range(len(task_list)):
        model, res = done_queue.get()
        model_res_combo = (model,)+res
        output_list.append(model_res_combo)

        if callback != None:
            print(f"{time.time()} Running callback!")
            callback(model_res_combo)

        print(f"{time.time():.1f} Finished running a task for model: {model}")
        print(f"{len(task_list)-len(output_list)} tasks remaining")

    # stop child processes
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    return output_list


def run_tasks(task_list, callback=None) -> list[tuple]:
    return run_tasks_parallel(task_list, 1, callable)


def load_or_run(name: str, tasks: list[tuple], force_run=False) -> list[tuple]:
    filename = f"{name}_{bad_hash_for_filename(tasks):.7g}"

    try:
        if force_run:
            raise Exception("force_run=True")
        
        loaded_data = np.load("./savedata/"+filename+".npy", allow_pickle=True)
        print(f"Loading of {name} succeeded!")
        return loaded_data
    except Exception as e:
        print(f"Failed loading of {name} because: " + str(e))
        res = run_tasks_parallel(tasks)

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

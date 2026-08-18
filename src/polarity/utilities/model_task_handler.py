import copy
import time
from multiprocessing import Process, Queue, cpu_count
from dataclasses import asdict

from polarity.model_enums import model_to_module

from pathlib import Path
DATA_DIR = Path(__file__).resolve().parents[3] / "results"


def worker(input, output):
    for model, args in iter(input.get, 'STOP'):
        label = args.pop('label', model.name)
        calc_ss_initial_condition = args.pop('calc_ss_initial_condition', False)
        print(f"{time.time():.1f} Running task with label: {label}")
        try:
            sol, setup = model_to_module(model).run_model(copy.deepcopy(args), 
                                                             calc_ss_initial_condition=calc_ss_initial_condition)
            output.put((label, (sol, asdict(setup))))
        except Exception as e:
            print(f"{time.time():.1f} Exception occurred while running task with label {label}; {e}")
            setup = model_to_module(model).Parameters(**args)
            output.put((label, ("FAILURE", asdict(setup))))


# Output of form {label: (sol, setup)}
def run_tasks_parallel(task_list, NUMBER_OF_PROCESSES=int(cpu_count()/1.5)) -> list[tuple[str, tuple]]:

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
        label, res = done_queue.get()
        output_list.append((label, res))

        print(f"{time.time():.1f} Finished running a task for label: {label}")
        print(f"{len(task_list)-len(output_list)} tasks remaining")

    # stop child processes
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    return output_list


def run_tasks(task_list) -> list[tuple[str, tuple]]:
    return run_tasks_parallel(task_list, 1)


# def load_or_run(filename: str, tasks: list[tuple], force_run=False) -> list[tuple]:
#     full_path = DATA_DIR / f"{filename}.npy"

#     try:
#         if force_run:
#             raise Exception("force_run=True")
        
#         loaded_data = np.load( DATA_DIR / f"{filename}.npy", allow_pickle=True)
#         print(f"Loading of {name} succeeded!")
#         return loaded_data
#     except Exception as e:
#         print(f"Failed loading of {name} because: " + str(e))
#         res = run_tasks_parallel(tasks)

#         print("Saving results")
#         np.save( DATA_DIR / f"{filename}.npy", res, allow_pickle=True)

#         return res

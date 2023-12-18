import time
from multiprocessing import Process, Queue, cpu_count
from models import model_to_module


def worker(input, output):
    for model, args in iter(input.get, 'STOP'):
        print(f"{time.time():.1f} Running task for model: {model}")
        try:
            result = model_to_module(model).run_model(args)
            output.put((model, result))
        except Exception as e:
            print(f"{time.time():.1f} Exception occurred while running task for model {model}; {e}")
            output.put((model, ("FAILURE",)))


# Output of form [(model, sol, kvals),...]
def run_tasks_parallel(task_list, NUMBER_OF_PROCESSES=cpu_count()-1):#int(cpu_count() // 1.5)):
    assert NUMBER_OF_PROCESSES >= 1

    # create queues
    task_queue = Queue()
    done_queue = Queue()

    # add tasks to queue
    for task in task_list:
        task_queue.put(task)

    # start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    output_list = []

    # get and handle results (unordered)
    for i in range(len(task_list)):
        model, res = done_queue.get()
        print(f"{time.time():.1f} Finished running a task for model: {model}")
        output_list.append((model,)+res)

    # stop child processes
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    return output_list


def run_tasks(task_list):
    return run_tasks_parallel(task_list, 1)

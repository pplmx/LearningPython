import os
from multiprocessing import Pool


def run_task_a(a: int):
    return a**2


def run_task_b(b: int):
    return b**3


def run_task_c(c: int):
    return c**4


result_list = []


def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)


def multi_proc(task_func_list, task_param_list, processes: int = os.cpu_count()):
    with Pool(processes=processes) as pool:
        # launching multiple evaluations asynchronously *may* use more processes
        for task_func, task_param in zip(task_func_list, task_param_list):
            print(f"task_func: {task_func}, task_params: {task_param}")
            pool.apply_async(task_func, task_param, callback=log_result)
        pool.close()
        pool.join()
        print(result_list)


if __name__ == "__main__":
    multi_tasks = [run_task_a, run_task_b, run_task_c]
    multi_tasks_params = [(1,), (2,), (3,)]
    multi_proc(multi_tasks, multi_tasks_params)

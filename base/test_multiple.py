import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool


# time cost wrapper
def time_cost(func):
    def wrapper(*args, **kwargs):
        begin = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} cost {round(end - begin, 2)} seconds')
        return res

    return wrapper


def f(x):
    # ======
    # Usually, test_multiprocessing_map is the fastest.
    # However, if uncomment the following line(time.sleep(1)), test_multiprocessing_map can be slowest.
    # You can try it: n = 100, proc_num = os.cpu_count(); then run main function.
    # You will find that test_multiprocessing_map is the slowest.
    # I don't know why.
    # But if you set n = 1000 or bigger, proc_num = os.cpu_count() or bigger;
    # you will find that test_multiprocessing_map becomes the fastest again.
    # ======
    # time.sleep(1)
    return x * x


@time_cost
def test_multiprocessing(task_num: int, processes: int = os.cpu_count()):
    # start a process pool with os.cpu_count() processes
    with Pool(processes=processes) as pool:
        # launching multiple evaluations asynchronously *may* use more processes
        # ======
        # NOTES:
        #       directly calling get() at each for-loop
        #       will cause the process pool to be blocked
        # ======
        # for i in range(task_num):
        #     pool.apply_async(f, (i,)).get()

        # first, apply a list of tasks without get()
        tasks_results = [pool.apply_async(f, (i,)) for i in range(task_num)]
        # collecting results
        _ = [r.get() for r in tasks_results]


@time_cost
def test_multiprocessing_map(task_num: int, processes: int = os.cpu_count()):
    with Pool(processes=processes) as pool:
        pool.map(f, range(task_num))


@time_cost
def test_concurrent_futures(task_num: int, processes: int = os.cpu_count()):
    with ProcessPoolExecutor(max_workers=processes) as executor:
        for i in range(task_num):
            executor.submit(f, i)


@time_cost
def test_concurrent_futures_map(task_num: int, processes: int = os.cpu_count()):
    with ProcessPoolExecutor(max_workers=processes) as executor:
        executor.map(f, range(task_num))


if __name__ == '__main__':
    n = 100000
    proc_num = os.cpu_count()
    print(f'Tasks = {n}, Processes = {proc_num}')
    test_multiprocessing(n, proc_num)
    test_multiprocessing_map(n, proc_num)
    # The following two solutions are not recommended, which is so slow.
    test_concurrent_futures_map(n, proc_num)
    # When n = 10000000, this function ......
    test_concurrent_futures(n, proc_num)

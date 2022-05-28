import concurrent
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
    print(x)
    time.sleep(1)
    return x * x


@time_cost
def test_multiprocessing():
    # start 4 worker processes
    with Pool(processes=os.cpu_count()) as pool:
        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = {i: pool.apply_async(f, (i,)) for i in range(100)}
        for i in multiple_results:
            try:
                print(f'number {i} squared is {multiple_results[i].get()}')
                # multiple_results[i].get()
            except Exception as e:
                print(f'number {i} failed, err: {e}')


@time_cost
def test_multiprocessing_map():
    with Pool(processes=os.cpu_count()) as pool:
        for i in pool.map(f, range(100)):
            print(f'Squared: {i}')


@time_cost
def test_concurrent_futures():
    # start cpu_count worker processes
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # launching multiple evaluations asynchronously *may* use more processes
        future_to_number = {executor.submit(f, i): i for i in range(100)}
        for future in concurrent.futures.as_completed(future_to_number):
            number = future_to_number[future]
            try:
                print(f'number {number} squared is {future.result()}')
                # future.result()
            except Exception as e:
                print(f'number {number} failed, err: {e}')


@time_cost
def test_concurrent_futures_map():
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in executor.map(f, range(100)):
            print(f'Squared: {i}')


if __name__ == '__main__':
    # test_multiprocessing()
    test_multiprocessing_map()
    # test_concurrent_futures()
    # test_concurrent_futures_map()

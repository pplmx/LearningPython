# time_cost with nanosecond
import time


def time_cost(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Cost: {end - start}s")
        return result

    return wrapper


# count the cost of func in nanosecond
def time_cost_ns(func):
    def wrapper(*args, **kwargs):
        start = time.time_ns()
        result = func(*args, **kwargs)
        end = time.time_ns()
        print(f"Cost: {end - start}ns")
        return result

    return wrapper

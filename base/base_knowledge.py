#!/usr/bin/env python
import functools
import time


class Apple:
    def __init__(self, color):
        self.__color = color


# 如果一个加上此装饰器的函数, 被调用三次; 则打印的数字会变成 3
def count_func_call(func):
    @functools.wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        wrapper_count_calls.num_calls += 1
        print(f"Call {wrapper_count_calls.num_calls} of {func.__name__!r}")
        return func(*args, **kwargs)

    wrapper_count_calls.num_calls = 0
    return wrapper_count_calls


# 所有加上此装饰器的函数, 每被调用一次, 打印的数字就会 +1
def count_all_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        count_all_calls.calls += 1
        print(f"{count_all_calls.calls} calls of {func.__name__!r}")
        return func(*args, **kwargs)

    count_all_calls.calls = 0
    return wrapper


def debug_truth(func):
    """Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)  # 3
        print(f"Calling {func.__name__}({signature})")
        ret = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {ret!r}")  # 4
        return ret

    return wrapper


# decorator without paras
def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Here is the function: {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


class DEBUG:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f"Here is the function: {self.func.__name__}")
        return self.func(*args, **kwargs)


# decorator with paras
def logging(level):
    def out_wrapper(func):
        def wrapper(*args, **kwargs):
            print(f"[{level}] - Function {func.__name__}() is running.")
            return func(*args, **kwargs)

        return wrapper

    return out_wrapper


class LOGGING:
    def __init__(self, level):
        self.level = level

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print(f"[{self.level}] - Function {func.__name__}() is running.")
            return func(*args, **kwargs)

        return wrapper


def func_cost(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} cost: {end - start}")
        return result

    return wrapper


def hello():
    print("hello world")


# @debug ==> debug(hello)
@debug
def hello1():
    print("hello world")


@DEBUG
def hello11():
    print("hello world")


# @logging('DEBUG') ==> logging('DEBUG')(hello)
@logging("DEBUG")
def hello2():
    print("hello world")


@LOGGING("DEBUG")
def hello22():
    print("hello world")


@func_cost
def hello3():
    print("hello world3")


@func_cost
def hello33(name):
    print(f"hello world, {name}")


if __name__ == "__main__":
    # hello1()
    # hello11()
    # hello2()
    # hello22()
    # hello3()
    # hello33('Yo')
    # red_apple = Apple("red")
    # logging('info')(hello33)("Tom")
    # debug(hello33)("Tom")
    # print(red_apple)

    # No Argument decorator
    hello = debug(hello)
    hello()

    # Argument decorator
    hello = logging("debug")(hello)
    hello()

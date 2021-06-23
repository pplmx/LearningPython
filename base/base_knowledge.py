#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Apple(object):

    def __init__(self, color):
        self.__color = color


# decorator without paras
def debug(func):
    def wrapper():
        print(f'Here is the function: {func.__name__}')
        return func()

    return wrapper


class DEBUG:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print(f'Here is the function: {self.func.__name__}')
        return self.func(*args, **kwargs)


# decorator with paras
def logging(level):
    def out_wrapper(func):
        def wrapper(*args, **kwargs):
            print(f'[{level}] - Function {func.__name__}() is running.')
            return func(*args, **kwargs)

        return wrapper

    return out_wrapper


class LOGGING:
    def __init__(self, level):
        self.level = level

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print(f'[{self.level}] - Function {func.__name__}() is running.')
            return func(*args, **kwargs)

        return wrapper


@debug
def hello1():
    print('hello world')


@DEBUG
def hello11():
    print('hello world')


@logging('DEBUG')
def hello2():
    print('hello world')


@LOGGING('DEBUG')
def hello22():
    print('hello world')


if __name__ == '__main__':
    hello1()
    hello11()
    hello2()
    hello22()
    red_apple = Apple("red")
    print(red_apple)

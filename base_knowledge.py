#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Apple(object):

    def __init__(self, color):
        self.__color = color


if __name__ == '__main__':
    red_apple = Apple("red")
    print(red_apple)

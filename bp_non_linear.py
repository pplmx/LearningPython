#!/usr/bin/env python


import numpy as np


def forward(x):
    a = x * x
    b = np.log(a)
    c = np.sqrt(b)
    return a, b, c


def backward(x, a, b, c, y):
    loss = c - y
    delta_c = loss
    delta_b = delta_c * 2 * np.sqrt(b)
    delta_a = delta_b * a
    delta_x = delta_a / 2 / x
    return loss, delta_x, delta_a, delta_b, delta_c

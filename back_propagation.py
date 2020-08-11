#!/usr/bin/env python

import numpy as np


def target_func(w, b):
    x = 2 * w + 3 * b
    y = 2 * b + 1
    z = x * y
    return x, y, z


def double_variable_right(w, b, target_z):
    error = 1e-5
    idx = 0
    while True:
        x, y, z = target_func(w, b)
        delta_z = z - target_z
        print(f'weight={w}, bias={b}, delta_z={delta_z}')
        if abs(delta_z) < error:
            break
        idx += 1
        delta_w = 0.5 * delta_z / (2 * y)
        delta_b = 0.5 * delta_z / (3 * y + 2 * x)
        print(f'delta_w={delta_w}, delta_b={delta_b}')
        w -= delta_w
        b -= delta_b

    print(f'The final weight = {w}, bias = {b}, Iterate times: {idx}')


if __name__ == '__main__':
    weight = 3
    bias = 4
    target = 150
    double_variable_right(weight, bias, target)

import jax.numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def relu_grad(x):
    return 1.0 * (x > 0)


def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2


def cross_entropy(y_hat, y):
    return -np.sum(y * np.log(y_hat))


def binary_cross_entropy(y_hat, y):
    return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


def mean_squared_error(y_hat, y):
    return np.mean((y_hat - y) ** 2)


def mean_absolute_error(y_hat, y):
    return np.mean(np.abs(y_hat - y))

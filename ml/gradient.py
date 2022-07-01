def f(x, y):
    return x ** 2 - y ** 2


# 梯度是一个向量
# 对于一个函数f(x, y)，它的梯度为: ▽f = grad f = <∂f/∂x, ∂f/∂y>
# 同理
# f(x, y, z)   的梯度为: ▽f = grad f = <∂f/∂x, ∂f/∂y, ∂f/∂z>
# f(x, y, z, w)的梯度为: ▽f = grad f = <∂f/∂x, ∂f/∂y, ∂f/∂z, ∂f/∂w>
# ...
def grad_f(x, y):
    return 2 * x, -2 * y


def gradient(func, x, y):
    return derivative_x(func, x, y), derivative_y(func, x, y)


# 这是函数 f(x, y) 在点 (x, y) 上的近似导数
def derivative_x(func, x, y):
    h = 1e-8
    return (func(x + h / 2, y) - func(x - h / 2, y)) / h


def derivative_y(func, x, y):
    h = 1e-8
    return (func(x, y + h / 2) - func(x, y - h / 2)) / h


if __name__ == '__main__':
    print(grad_f(144, 10))
    print(gradient(f, 144, 10))

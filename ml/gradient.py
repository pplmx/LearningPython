import numpy as np
from matplotlib import cm, rcParams
from matplotlib import pyplot as plt

"""
这是一个学习梯度(gradient)计算的示例
梯度在机器学习中非常重要,它告诉我们函数值变化最快的方向

理论知识:
1. 梯度是一个向量,它指向函数值增长最快的方向
2. 对于函数f(x,y),梯度包含两个部分:
   - 对x的偏导数: ∂f/∂x
   - 对y的偏导数: ∂f/∂y
3. 写作数学符号是: ▽f = <∂f/∂x, ∂f/∂y>
"""

# 设置中文字体
rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]  # 用黑体显示中文
rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def f(x, y):
    """
    这是我们要分析的二元函数: f(x,y) = x² - y²
    参数:
        x: 第一个变量
        y: 第二个变量
    返回:
        函数值: x的平方减去y的平方
    """
    return x**2 - y**2


def grad_f(x, y):
    """
    计算函数f在点(x,y)处的梯度
    这是通过数学推导得到的精确梯度

    对f(x,y) = x² - y²求偏导:
    ∂f/∂x = 2x
    ∂f/∂y = -2y
    """
    return 2 * x, -2 * y


def gradient(func, x, y):
    """
    通过数值方法计算任意函数在点(x,y)处的梯度
    这种方法可以用于那些难以求出解析解的函数
    """
    return derivative_x(func, x, y), derivative_y(func, x, y)


def derivative_x(func, x, y):
    """
    计算函数关于x的偏导数
    使用中心差分法(central difference)近似导数:
    f'(x) ≈ [f(x+h/2) - f(x-h/2)] / h
    """
    h = 1e-8  # 一个很小的数,用于近似计算
    return (func(x + h / 2, y) - func(x - h / 2, y)) / h


def derivative_y(func, x, y):
    """
    计算函数关于y的偏导数
    方法同上
    """
    h = 1e-8
    return (func(x, y + h / 2) - func(x, y - h / 2)) / h


def plot_function_and_gradient():
    # 创建网格点
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # 创建一个包含两个子图的图形
    fig = plt.figure(figsize=(15, 6))

    # 1. 3D表面图
    ax1 = fig.add_subplot(121, projection="3d")
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("f(x,y)")
    ax1.set_title("函数f(x,y) = x² - y²的3D图像")
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # 2. 等高线图和梯度向量
    ax2 = fig.add_subplot(122)
    # 绘制等高线
    contours = ax2.contour(X, Y, Z, levels=20, cmap="coolwarm")
    ax2.clabel(contours, inline=True, fontsize=8)

    # 在选定点绘制梯度向量
    points = [(2, 1), (-1, -2), (0, 2), (1, -1)]
    for point in points:
        x, y = point
        dx, dy = grad_f(x, y)  # 获取梯度
        # 归一化梯度向量使其长度合适
        length = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / length, dy / length
        # 绘制箭头
        ax2.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.1,
            head_length=0.2,
            fc="red",
            ec="red",
            length_includes_head=True,
        )

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("等高线图和梯度向量")
    ax2.grid(True)
    ax2.axis("equal")

    plt.tight_layout()
    plt.show()


def plot_gradient_field():
    """绘制梯度场"""
    # 创建较稀疏的网格点用于绘制梯度场
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    X, Y = np.meshgrid(x, y)

    # 计算每个点的梯度
    U = 2 * X  # ∂f/∂x = 2x
    V = -2 * Y  # ∂f/∂y = -2y

    # 计算梯度场的强度
    norm = np.sqrt(U**2 + V**2)

    plt.figure(figsize=(8, 8))
    # 绘制背景等高线
    Z = f(X, Y)
    plt.contour(X, Y, Z, levels=20, cmap="coolwarm", alpha=0.3)

    # 绘制梯度场
    plt.quiver(
        X,
        Y,
        U / norm,
        V / norm,
        norm,
        cmap="viridis",
        angles="xy",
        scale_units="xy",
        scale=20,
    )

    plt.colorbar(label="梯度强度")
    plt.title("函数的梯度场")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    # 1. 首先展示函数图像和部分梯度向量
    plot_function_and_gradient()

    # 2. 然后展示整个梯度场
    plot_gradient_field()

    # 3. 验证某点的精确梯度和数值梯度
    x, y = 144, 10
    print(f"\n在点({x}, {y})处:")
    print(f"解析解(精确梯度): {grad_f(x, y)}")
    print(f"数值解(近似梯度): {gradient(f, x, y)}")

#!/usr/bin/env python
"""
梯度下降算法示例 - 双变量优化问题

这个示例展示了如何使用梯度下降算法来解决一个简单的优化问题:
给定目标值target_z,通过调整权重w和偏置b,使得函数输出z逼近目标值。

核心概念:
1. 损失函数: z与目标值target_z之间的差异(delta_z)
2. 梯度下降: 计算损失函数对各参数的偏导数,并沿着梯度反方向更新参数
3. 学习率: 控制每次参数更新的步长(本例中为0.5)
4. 收敛条件: 当损失值小于某个阈值时停止迭代

运行示例:
    python bp_linear.py
"""

import matplotlib.pyplot as plt

from utils.common import time_cost_ns


def target_func(w: float, b: float) -> tuple[float, float, float]:
    """目标函数: 计算给定参数下的x, y, z值

    函数关系:
        x = 2w + 3b
        y = 2b + 1
        z = x * y

    Args:
        w: 权重参数
        b: 偏置参数

    Returns:
        tuple: (x, y, z) 计算结果
    """
    x = 2 * w + 3 * b  # 计算中间变量x
    y = 2 * b + 1  # 计算中间变量y
    z = x * y  # 计算最终输出z
    return x, y, z


class GradientDescentOptimizer:
    """梯度下降优化器

    用于记录优化过程中的状态变化,方便后续分析和可视化
    """

    def __init__(self):
        self.w_history = []  # 记录权重变化
        self.b_history = []  # 记录偏置变化
        self.z_history = []  # 记录输出值变化

    def record_state(self, w: float, b: float, z: float):
        """记录当前迭代的状态"""
        self.w_history.append(w)
        self.b_history.append(b)
        self.z_history.append(z)

    def plot_optimization_process(self, target_z: float):
        """可视化优化过程

        绘制三张图:
        1. 参数w和b的变化曲线
        2. 输出值z逼近目标值的过程
        3. 参数空间中的优化路径
        """
        plt.figure(figsize=(15, 5))

        # 绘制参数变化
        plt.subplot(131)
        plt.plot(self.w_history, label="weight (w)")
        plt.plot(self.b_history, label="bias (b)")
        plt.title("Parameter Changes")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.legend()

        # 绘制输出值变化
        plt.subplot(132)
        plt.plot(self.z_history, label="current z")
        plt.axhline(y=target_z, color="r", linestyle="--", label="target z")
        plt.title("Output Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("z value")
        plt.legend()

        # 绘制参数空间优化路径
        plt.subplot(133)
        plt.plot(self.w_history, self.b_history, "b.-")
        plt.plot(self.w_history[0], self.b_history[0], "go", label="start")
        plt.plot(self.w_history[-1], self.b_history[-1], "ro", label="end")
        plt.title("Optimization Path")
        plt.xlabel("weight (w)")
        plt.ylabel("bias (b)")
        plt.legend()

        plt.tight_layout()
        plt.show()


@time_cost_ns
def optimize_parameters(
    w: float,
    b: float,
    target_z: float,
    learning_rate: float = 0.5,
    error_threshold: float = 1e-5,
    max_iterations: int = 1000,
) -> tuple[float, float, float, int]:
    """使用梯度下降优化参数

    Args:
        w: 初始权重值
        b: 初始偏置值
        target_z: 目标输出值
        learning_rate: 学习率,控制参数更新步长
        error_threshold: 误差阈值,当误差小于此值时停止迭代
        max_iterations: 最大迭代次数,防止无限循环

    Returns:
        tuple: (最终权重, 最终偏置, 最终误差, 迭代次数)
    """
    optimizer = GradientDescentOptimizer()
    iteration = 0

    while True:
        # 1. 前向计算
        x, y, z = target_func(w, b)
        delta_z = z - target_z  # 计算误差

        # 记录当前状态
        optimizer.record_state(w, b, z)
        print(f"Iteration {iteration}:")
        print(f"  w={w:.6f}, b={b:.6f}, z={z:.6f}, Δz={delta_z:.6f}")

        # 2. 检查停止条件
        if abs(delta_z) < error_threshold:
            print("\n✓ 优化成功!")
            break

        if iteration >= max_iterations:
            print("\n! 达到最大迭代次数,优化未收敛")
            break

        # 3. 计算梯度 (对w和b的偏导数)
        delta_w = learning_rate * delta_z / (2 * y)  # ∂z/∂w
        delta_b = learning_rate * delta_z / (3 * y + 2 * x)  # ∂z/∂b
        print(f"  Δw={delta_w:.6f}, Δb={delta_b:.6f}\n")

        # 4. 更新参数 (沿梯度反方向)
        w -= delta_w
        b -= delta_b
        iteration += 1

    # 输出最终结果
    print(f"""
最终结果:
    权重 w = {w:.6f}
    偏置 b = {b:.6f}
    误差 Δz = {delta_z:.6f}
    迭代次数: {iteration}
    """)

    # 可视化优化过程
    optimizer.plot_optimization_process(target_z)

    return w, b, delta_z, iteration


def demo():
    """主函数：设置初始参数并运行优化"""
    # 设置初始参数
    initial_w = 3.0  # 初始权重
    initial_b = 4.0  # 初始偏置
    target_z = 150.0  # 目标值

    print(f"""
开始优化:
    初始权重 w = {initial_w}
    初始偏置 b = {initial_b}
    目标值 z = {target_z}
    """)

    # 运行优化
    optimize_parameters(initial_w, initial_b, target_z)


if __name__ == "__main__":
    demo()

#!/usr/bin/env python
"""
非线性优化示例 - 使用梯度下降优化非线性函数

这个示例展示了如何使用梯度下降算法来优化一个非线性函数:
f(x, y) = sin(x^2/4 + y^2/2) + 2*exp(-(x^2 + y^2)/8) + (x^2 + 2*y^2)/10

目标是找到函数的局部最小值。

核心概念:
1. 非线性函数：包含正弦、指数等非线性项
2. 偏导数：计算每个变量的梯度
3. 学习率衰减：动态调整学习率以提高收敛性
4. 局部最小值：函数可能存在多个局部最小值

运行示例:
    python nonlinear_optimization.py
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


class NonlinearFunction:
    """非线性函数实现类"""

    @staticmethod
    def evaluate(x: float, y: float) -> float:
        """计算函数值
        f(x, y) = sin(x^2/4 + y^2/2) + 2*exp(-(x^2 + y^2)/8) + (x^2 + 2*y^2)/10
        """
        term1 = np.sin(x**2 / 4 + y**2 / 2)
        term2 = 2 * np.exp(-(x**2 + y**2) / 8)
        term3 = (x**2 + 2 * y**2) / 10
        return term1 + term2 + term3

    @staticmethod
    def gradient(x: float, y: float) -> Tuple[float, float]:
        """计算函数关于x和y的偏导数

        ∂f/∂x = (x/2)*cos(x^2/4 + y^2/2) - (x/4)*2*exp(-(x^2 + y^2)/8) + x/5
        ∂f/∂y = y*cos(x^2/4 + y^2/2) - (y/4)*2*exp(-(x^2 + y^2)/8) + 2*y/5
        """
        common_term1 = np.cos(x**2 / 4 + y**2 / 2)
        common_term2 = np.exp(-(x**2 + y**2) / 8)

        dx = (x / 2) * common_term1 - (x / 4) * 2 * common_term2 + x / 5
        dy = y * common_term1 - (y / 4) * 2 * common_term2 + 2 * y / 5

        return dx, dy


class OptimizationVisualizer:
    """优化过程可视化器"""

    def __init__(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        self.x_range = x_range
        self.y_range = y_range
        self.history = []

        # 创建网格数据
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.vectorize(NonlinearFunction.evaluate)(self.X, self.Y)

    def add_point(self, x: float, y: float, z: float):
        """记录优化过程中的点"""
        self.history.append((x, y, z))

    def plot_surface(self):
        """绘制3D曲面图和优化路径"""
        fig = plt.figure(figsize=(15, 5))

        # 3D曲面图
        ax1 = fig.add_subplot(121, projection="3d")
        surf = ax1.plot_surface(self.X, self.Y, self.Z, cmap="viridis", alpha=0.8)

        # 绘制优化路径
        path = np.array(self.history)
        ax1.plot(
            path[:, 0],
            path[:, 1],
            path[:, 2],
            "r.-",
            linewidth=2,
            label="Optimization Path",
        )

        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")
        ax1.set_title("Function Surface and Optimization Path")

        # 等高线图
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(self.X, self.Y, self.Z, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)

        # 绘制优化路径
        ax2.plot(path[:, 0], path[:, 1], "r.-", linewidth=2, label="Optimization Path")
        ax2.scatter(path[0, 0], path[0, 1], color="g", s=100, label="Start")
        ax2.scatter(path[-1, 0], path[-1, 1], color="r", s=100, label="End")

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title("Contour Map and Optimization Path")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def create_animation(self):
        """创建优化过程动画"""
        fig, ax = plt.subplots(figsize=(8, 8))
        contour = ax.contour(self.X, self.Y, self.Z, levels=20)
        ax.clabel(contour, inline=True, fontsize=8)

        path = np.array(self.history)
        (line,) = ax.plot([], [], "r.-", linewidth=2)
        (point,) = ax.plot([], [], "ro", markersize=10)

        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_title("Optimization Process Animation")

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate(i):
            line.set_data(path[: i + 1, 0], path[: i + 1, 1])
            point.set_data(path[i : i + 1, 0], path[i : i + 1, 1])
            return line, point

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(path),
            interval=100,
            blit=True,
            repeat=False,
        )
        plt.show()


class GradientDescentOptimizer:
    """梯度下降优化器"""

    def __init__(
        self,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        # 可视化器
        self.visualizer = OptimizationVisualizer((-4, 4), (-4, 4))

    def optimize(self, x0: float, y0: float) -> Tuple[float, float, float, List]:
        """执行梯度下降优化

        Args:
            x0: x的初始值
            y0: y的初始值

        Returns:
            Tuple[float, float, float, List]: (最优x值, 最优y值, 最优函数值, 收敛历史)
        """
        x, y = x0, y0
        vx, vy = 0, 0  # 动量项

        # 记录收敛历史
        history = []

        print("\n开始优化:")
        for i in range(self.max_iterations):
            # 计算当前值
            z = NonlinearFunction.evaluate(x, y)
            self.visualizer.add_point(x, y, z)
            history.append((x, y, z))

            # 计算梯度
            dx, dy = NonlinearFunction.gradient(x, y)

            # 检查收敛
            if np.sqrt(dx**2 + dy**2) < self.tolerance:
                print(f"\n✓ 优化收敛于第{i + 1}次迭代!")
                break

            # 更新动量
            vx = self.momentum * vx - self.learning_rate * dx
            vy = self.momentum * vy - self.learning_rate * dy

            # 更新参数
            x += vx
            y += vy

            # 打印进度
            if i % 10 == 0:
                print(
                    f"迭代 {i:4d}: x={x:8.4f}, y={y:8.4f}, z={z:8.4f}, "
                    f"梯度范数={np.sqrt(dx ** 2 + dy ** 2):8.4f}"
                )

        final_z = NonlinearFunction.evaluate(x, y)
        print(f"""
最终结果:
    x = {x:.6f}
    y = {y:.6f}
    f(x,y) = {final_z:.6f}
    迭代次数: {i + 1}
        """)

        return x, y, final_z, history

    def visualize_optimization(self):
        """可视化优化过程"""
        print("\n正在生成可视化...")
        self.visualizer.plot_surface()
        self.visualizer.create_animation()


def demo():
    """主函数：运行优化示例"""
    # 设置随机初始点
    np.random.seed(42)
    x0 = np.random.uniform(-4, 4)
    y0 = np.random.uniform(-4, 4)

    print(f"""
非线性函数优化示例
------------------
函数形式:
    f(x, y) = sin(x^2/4 + y^2/2) + 2*exp(-(x^2 + y^2)/8) + (x^2 + 2*y^2)/10

初始参数:
    x0 = {x0:.4f}
    y0 = {y0:.4f}
    初始函数值 = {NonlinearFunction.evaluate(x0, y0):.4f}
    """)

    # 创建优化器并运行
    optimizer = GradientDescentOptimizer(
        learning_rate=0.1, momentum=0.9, max_iterations=500, tolerance=1e-6
    )

    # 执行优化
    optimizer.optimize(x0, y0)

    # 显示可视化结果
    optimizer.visualize_optimization()


if __name__ == "__main__":
    demo()

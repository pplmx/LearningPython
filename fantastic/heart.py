import random
import tkinter as tk
from dataclasses import dataclass
from math import cos, log, pi, sin
from tkinter import messagebox
from tkinter.constants import CENTER
from typing import Dict, List, Set, Tuple

# 配置常量
CONFIG = {
    "WINDOW_WIDTH": 888,
    "WINDOW_HEIGHT": 500,
    "HEART_COLOR": "#FF69B4",  # 更鲜艳的粉色
    "HEART_SCALE": 11,
    "BACKGROUND_COLOR": "#000000",
    "TEXT_COLOR": "#FF1493",  # 深粉色
    "FONT_FAMILY": "Helvetica",
    "ANIMATION_SPEED": 120,  # 更流畅的动画速度
    "DIALOG_WIDTH": 300,
    "DIALOG_HEIGHT": 120,
    "MAX_CLOSE_ATTEMPTS": 3,  # 最大关闭尝试次数
}


@dataclass
class Point:
    x: float
    y: float
    size: int = 1


class HeartAnimation:
    def __init__(self, frames: int = 20):
        self.heart_x = CONFIG["WINDOW_WIDTH"] / 2
        self.heart_y = CONFIG["WINDOW_HEIGHT"] / 2
        self.frames = frames
        self.points: Set[Tuple[float, float]] = set()
        self.edge_points: Set[Tuple[float, float]] = set()
        self.center_points: Set[Tuple[float, float]] = set()
        self.frame_points: Dict[int, List[Point]] = {}

        self._initialize_points()
        self._generate_frames()

    def _heart_function(
        self, t: float, scale: float = CONFIG["HEART_SCALE"]
    ) -> Tuple[int, int]:
        """生成心形曲线的点"""
        x = 16 * (sin(t) ** 3)
        y = -(13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t))
        x = x * scale + self.heart_x
        y = y * scale + self.heart_y
        return int(x), int(y)

    def _scatter_point(
        self, x: float, y: float, beta: float = 0.15
    ) -> Tuple[float, float]:
        """散射效果"""
        ratio_x = -beta * log(random.random())
        ratio_y = -beta * log(random.random())
        dx = ratio_x * (x - self.heart_x)
        dy = ratio_y * (y - self.heart_y)
        return x - dx, y - dy

    def _shrink_point(self, x: float, y: float, ratio: float) -> Tuple[float, float]:
        """收缩效果"""
        force = -1 / (((x - self.heart_x) ** 2 + (y - self.heart_y) ** 2) ** 0.6)
        dx = ratio * force * (x - self.heart_x)
        dy = ratio * force * (y - self.heart_y)
        return x - dx, y - dy

    def _initialize_points(self):
        """初始化所有点集"""
        # 生成基础心形点
        for _ in range(2000):
            t = random.uniform(0, 2 * pi)
            self.points.add(self._heart_function(t))

        # 生成边缘扩散效果
        for x, y in list(self.points):
            for _ in range(3):
                self.edge_points.add(self._scatter_point(x, y, 0.05))

        # 生成中心扩散效果
        point_list = list(self.points)
        for _ in range(4000):
            x, y = random.choice(point_list)
            self.center_points.add(self._scatter_point(x, y, 0.17))

    def _generate_frames(self):
        """生成动画帧"""
        for frame in range(self.frames):
            self._calculate_frame(frame)

    def _calculate_frame(self, frame: int):
        """计算每一帧的点位置"""
        ratio = 10 * self._curve(frame / 10 * pi)
        halo_radius = int(4 + 6 * (1 + self._curve(frame / 10 * pi)))
        halo_number = int(3000 + 4000 * abs(self._curve(frame / 10 * pi) ** 2))

        frame_points = []
        heart_halo = set()

        # 生成光晕效果
        for _ in range(halo_number):
            t = random.uniform(0, 2 * pi)
            x, y = self._heart_function(t, 11.6)
            x, y = self._shrink_point(x, y, halo_radius)

            if (x, y) not in heart_halo:
                heart_halo.add((x, y))
                x += random.randint(-14, 14)
                y += random.randint(-14, 14)
                frame_points.append(Point(x, y, random.choice((1, 2, 2))))

        # 添加所有其他点
        for point_set, size_range in [
            (self.points, (1, 3)),
            (self.edge_points, (1, 2)),
            (self.center_points, (1, 2)),
        ]:
            for x, y in point_set:
                x, y = self._calculate_position(x, y, ratio)
                frame_points.append(Point(x, y, random.randint(*size_range)))

        self.frame_points[frame] = frame_points

    def _calculate_position(
        self, x: float, y: float, ratio: float
    ) -> Tuple[float, float]:
        """计算点的新位置"""
        force = 1 / (((x - self.heart_x) ** 2 + (y - self.heart_y) ** 2) ** 0.520)
        dx = ratio * force * (x - self.heart_x) + random.randint(-1, 1)
        dy = ratio * force * (y - self.heart_y) + random.randint(-1, 1)
        return x - dx, y - dy

    @staticmethod
    def _curve(p: float) -> float:
        """生成曲线效果"""
        return 2 * (2 * sin(4 * p)) / (2 * pi)

    def render(self, canvas: tk.Canvas, frame: int):
        """渲染当前帧"""
        canvas.delete("all")
        for point in self.frame_points[frame % self.frames]:
            canvas.create_rectangle(
                point.x,
                point.y,
                point.x + point.size,
                point.y + point.size,
                width=0,
                fill=CONFIG["HEART_COLOR"],
            )

        # 添加闪烁效果
        if random.random() < 0.1:  # 10%概率产生闪烁
            x = random.randint(0, CONFIG["WINDOW_WIDTH"])
            y = random.randint(0, CONFIG["WINDOW_HEIGHT"])
            size = random.randint(1, 3)
            canvas.create_oval(x, y, x + size, y + size, fill="white", width=0)


class HeartApp:
    def __init__(self):
        self.root = tk.Tk()
        self.close_attempts = 0
        self._setup_main_window()
        self.heart_animation = None

    def _setup_main_window(self):
        """设置主窗口"""
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - CONFIG["DIALOG_WIDTH"]) // 2
        y = (screen_height - CONFIG["DIALOG_HEIGHT"]) // 2

        self.root.geometry(
            f"{CONFIG['DIALOG_WIDTH']}x{CONFIG['DIALOG_HEIGHT']}+{x}+{y}"
        )
        self.root.title("❤ Love Message ❤")
        self.root.resizable(False, False)
        self.root.configure(bg="#FFE4E1")  # 浅粉色背景

        # 创建渐变效果的标题
        title_frame = tk.Frame(self.root, bg="#FFE4E1")
        title_frame.pack(pady=10)
        tk.Label(
            title_frame,
            text="亲爱的，做我女朋友好吗？",
            font=("微软雅黑", 14, "bold"),
            fg="#FF1493",
            bg="#FFE4E1",
        ).pack()

        # 创建按钮框架
        button_frame = tk.Frame(self.root, bg="#FFE4E1")
        button_frame.pack(pady=15)

        # 美化按钮样式
        button_style = {
            "width": 8,
            "height": 1,
            "font": ("微软雅黑", 10),
            "relief": "raised",
            "borderwidth": 2,
        }

        # 创建三个按钮：同意、考虑一下、退出
        tk.Button(
            button_frame,
            text="好呀 ❤",
            command=self._on_accept,
            bg="#FF69B4",
            fg="white",
            **button_style,
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            button_frame,
            text="考虑一下",
            command=self._on_reject,
            bg="#FFB6C1",
            fg="white",
            **button_style,
        ).pack(side=tk.LEFT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _create_heart_window(self):
        """创建心形动画窗口"""
        heart_window = tk.Toplevel(self.root)
        screen_width = heart_window.winfo_screenwidth()
        screen_height = heart_window.winfo_screenheight()
        x = (screen_width - CONFIG["WINDOW_WIDTH"]) // 2
        y = (screen_height - CONFIG["WINDOW_HEIGHT"]) // 2

        heart_window.geometry(
            f"{CONFIG['WINDOW_WIDTH']}x{CONFIG['WINDOW_HEIGHT']}+{x}+{y}"
        )
        heart_window.title("❤ Love You ❤")
        heart_window.resizable(False, False)

        canvas = tk.Canvas(
            heart_window,
            width=CONFIG["WINDOW_WIDTH"],
            height=CONFIG["WINDOW_HEIGHT"],
            bg=CONFIG["BACKGROUND_COLOR"],
            highlightthickness=0,
        )
        canvas.pack()

        # 添加文字标签
        label = tk.Label(
            heart_window,
            text="I Love You!",
            bg=CONFIG["BACKGROUND_COLOR"],
            fg=CONFIG["TEXT_COLOR"],
            font=(CONFIG["FONT_FAMILY"], 25, "bold"),
        )
        label.place(relx=0.5, rely=0.5, anchor=CENTER)

        self.heart_animation = HeartAnimation()
        self._animate(heart_window, canvas, 0)

        # 设置心形窗口的关闭事件
        heart_window.protocol("WM_DELETE_WINDOW", self.root.quit)

    def _animate(self, window: tk.Tk, canvas: tk.Canvas, frame: int):
        """动画循环"""
        self.heart_animation.render(canvas, frame)
        window.after(
            CONFIG["ANIMATION_SPEED"], self._animate, window, canvas, frame + 1
        )

    def _on_accept(self):
        """接受按钮回调"""
        self.root.withdraw()  # 隐藏主窗口
        self._create_heart_window()

    def _on_reject(self):
        """拒绝按钮回调"""
        messages = [
            "再考虑一下呗～",
            "你确定要拒绝这么可爱的我吗？",
            "人家会很难过的～",
            "给我一个机会嘛～",
            "你真的忍心拒绝我吗？",
        ]
        messagebox.showinfo("❤ Sweet Message ❤", random.choice(messages))

    def _on_closing(self):
        """窗口关闭回调"""
        self.close_attempts += 1
        if self.close_attempts >= CONFIG["MAX_CLOSE_ATTEMPTS"]:
            if messagebox.askyesno("确认退出", "真的要退出吗？"):
                self.root.quit()
        else:
            remaining = CONFIG["MAX_CLOSE_ATTEMPTS"] - self.close_attempts
            messagebox.showinfo("❤ Sweet Message ❤", f"再给我{remaining}次机会好不好～")

    def run(self):
        """运行应用"""
        self.root.mainloop()


if __name__ == "__main__":
    app = HeartApp()
    app.run()

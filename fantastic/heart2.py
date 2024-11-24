import random
import tkinter as tk
from dataclasses import dataclass, field
from math import cos, pi, sin
from tkinter import messagebox
from tkinter.constants import CENTER
from typing import List

# 全局配置常量
CONFIG = {
    # 窗口设置
    "WINDOW_WIDTH": 1000,  # 更大的窗口尺寸
    "WINDOW_HEIGHT": 600,
    "DIALOG_WIDTH": 400,  # 更宽的对话框
    "DIALOG_HEIGHT": 180,
    # 颜色设置
    "PRIMARY_COLOR": "#FF69B4",  # 主要粉色
    "SECONDARY_COLOR": "#FF1493",  # 次要深粉色
    "BACKGROUND_COLOR": "#000000",  # 深邃星空背景
    "STAR_COLORS": ["#FFE4E1", "#FFF0F5", "#FFB6C1"],  # 星星颜色组
    # 心形设置
    "HEART_SCALE": 12,  # 更大的心形
    "HEART_COLORS": [  # 渐变色彩组
        "#FF69B4",
        "#FF1493",
        "#DB7093",
        "#FFB6C1",
        "#FFC0CB",
        "#FF69B4",
        "#FF00FF",
    ],
    # 动画设置
    "ANIMATION_SPEED": 100,  # 动画速度
    "PULSE_SPEED": 0.1,  # 心跳速度
    "STAR_DENSITY": 50,  # 星星密度
    "PARTICLE_COUNT": 2500,  # 粒子数量
    # 交互设置
    "MAX_CLOSE_ATTEMPTS": 5,  # 增加关闭尝试次数
    # 字体设置
    "FONT_FAMILY": "微软雅黑",
    "MESSAGE_FONT_SIZE": 16,
    "BUTTON_FONT_SIZE": 12,
}


@dataclass
class Particle:
    """粒子类 - 用于创建动画效果的基本单位"""

    x: float
    y: float
    size: int = 1
    color: str = field(default_factory=lambda: CONFIG["PRIMARY_COLOR"])
    velocity: float = 0
    angle: float = 0


class Star:
    """星星类 - 用于创建背景闪烁效果"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.size = random.uniform(0.5, 2)
        self.twinkle_speed = random.uniform(0.02, 0.05)
        self.phase = random.uniform(0, 2 * pi)
        self.color = random.choice(CONFIG["STAR_COLORS"])
        self.brightness = None

    def update(self, time: float) -> None:
        """更新星星的闪烁状态"""
        self.brightness = (sin(self.phase + time * self.twinkle_speed) + 1) / 2


class HeartAnimation:
    """心形动画类 - 处理主要的动画逻辑"""

    def __init__(self, frames: int = 30):
        self.heart_x = CONFIG["WINDOW_WIDTH"] / 2
        self.heart_y = CONFIG["WINDOW_HEIGHT"] / 2
        self.frames = frames
        self.particles: List[Particle] = []
        self.stars: List[Star] = []
        self.time = 0
        self.pulse_phase = 0

        # 初始化动画元素
        self._initialize_stars()
        self._initialize_particles()

    def _initialize_stars(self) -> None:
        """初始化背景星星"""
        for _ in range(CONFIG["STAR_DENSITY"]):
            x = random.uniform(0, CONFIG["WINDOW_WIDTH"])
            y = random.uniform(0, CONFIG["WINDOW_HEIGHT"])
            self.stars.append(Star(x, y))

    def _initialize_particles(self) -> None:
        """初始化心形粒子"""
        for _ in range(CONFIG["PARTICLE_COUNT"]):
            t = random.uniform(0, 2 * pi)
            particle = self._create_heart_particle(t)
            self.particles.append(particle)

    def _create_heart_particle(self, t: float) -> Particle:
        """创建一个心形轮廓上的粒子"""
        # 心形方程
        x = 16 * (sin(t) ** 3)
        y = -(13 * cos(t) - 5 * cos(2 * t) - 2 * cos(3 * t) - cos(4 * t))

        # 应用缩放和位置偏移
        scale = CONFIG["HEART_SCALE"] * (1 + random.uniform(-0.1, 0.1))
        x = x * scale + self.heart_x
        y = y * scale + self.heart_y

        # 添加随机颜色变化
        color = random.choice(CONFIG["HEART_COLORS"])

        return Particle(x, y, random.randint(1, 3), color)

    def _update_particles(self) -> None:
        """更新所有粒子的状态"""
        pulse_factor = 1 + 0.1 * sin(self.pulse_phase)

        for particle in self.particles:
            # 添加脉动效果
            dx = (particle.x - self.heart_x) * pulse_factor
            dy = (particle.y - self.heart_y) * pulse_factor

            # 更新粒子位置
            particle.x = self.heart_x + dx
            particle.y = self.heart_y + dy

            # 随机颜色变化
            if random.random() < 0.01:
                particle.color = random.choice(CONFIG["HEART_COLORS"])

    def render(self, canvas: tk.Canvas) -> None:
        """渲染当前帧"""
        canvas.delete("all")

        # 渲染星空背景
        self._render_stars(canvas)

        # 渲染心形粒子
        self._render_particles(canvas)

        # 更新动画状态
        self.time += 0.1
        self.pulse_phase += CONFIG["PULSE_SPEED"]
        self._update_particles()

    def _render_stars(self, canvas: tk.Canvas) -> None:
        """渲染背景星星"""
        for star in self.stars:
            star.update(self.time)
            canvas.create_oval(
                star.x - star.size,
                star.y - star.size,
                star.x + star.size,
                star.y + star.size,
                fill=star.color,
                width=0,
                stipple="gray50",  # 添加朦胧效果
            )

    def _render_particles(self, canvas: tk.Canvas) -> None:
        """渲染心形粒子"""
        for particle in self.particles:
            canvas.create_rectangle(
                particle.x,
                particle.y,
                particle.x + particle.size,
                particle.y + particle.size,
                width=0,
                fill=particle.color,
            )


class HeartApp:
    """主应用程序类"""

    def __init__(self):
        self.root = tk.Tk()
        self.close_attempts = 0
        self._setup_main_window()
        self.heart_animation = None

        # 浪漫消息列表
        self.love_messages = [
            "你是我生命中最璀璨的星光✨",
            "每一次心跳都为你跳动❤",
            "你是我最美好的遇见🌹",
            "想要和你看遍世界的每个角落🌎",
            "愿意为你写下最浪漫的诗句📝",
            "你是我最甜蜜的心动瞬间💕",
        ]

    def _setup_main_window(self) -> None:
        """设置主窗口"""
        # 窗口居中显示
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - CONFIG["DIALOG_WIDTH"]) // 2
        y = (screen_height - CONFIG["DIALOG_HEIGHT"]) // 2

        self.root.geometry(
            f"{CONFIG['DIALOG_WIDTH']}x{CONFIG['DIALOG_HEIGHT']}+{x}+{y}"
        )
        self.root.title("❤ Love Message ❤")
        self.root.resizable(False, False)

        # 设置渐变背景
        self._create_gradient_background()

        # 创建主要内容
        self._create_content()

    def _create_gradient_background(self) -> None:
        """创建渐变背景效果"""
        background = tk.Canvas(
            self.root,
            width=CONFIG["DIALOG_WIDTH"],
            height=CONFIG["DIALOG_HEIGHT"],
            highlightthickness=0,
        )
        background.pack(fill="both", expand=True)

        # 创建渐变色彩
        for i in range(CONFIG["DIALOG_HEIGHT"]):
            r = int(255 - (i / CONFIG["DIALOG_HEIGHT"]) * 30)
            g = int(228 - (i / CONFIG["DIALOG_HEIGHT"]) * 20)
            b = int(225 - (i / CONFIG["DIALOG_HEIGHT"]) * 20)
            color = f"#{r:02x}{g:02x}{b:02x}"
            background.create_line(0, i, CONFIG["DIALOG_WIDTH"], i, fill=color)

    def _create_content(self) -> None:
        """创建窗口内容"""
        # 标题框架
        title_frame = tk.Frame(self.root, bg="#FFE4E1")
        title_frame.place(relx=0.5, rely=0.2, anchor=CENTER)

        # 标题文本
        tk.Label(
            title_frame,
            text="亲爱的，做我女朋友好吗？",
            font=(CONFIG["FONT_FAMILY"], CONFIG["MESSAGE_FONT_SIZE"], "bold"),
            fg=CONFIG["SECONDARY_COLOR"],
            bg="#FFE4E1",
        ).pack()

        # 按钮框架
        button_frame = tk.Frame(self.root, bg="#FFE4E1")
        button_frame.place(relx=0.5, rely=0.6, anchor=CENTER)

        # 创建按钮
        self._create_buttons(button_frame)

    def _create_buttons(self, frame: tk.Frame) -> None:
        """创建按钮"""
        button_style = {
            "width": 12,
            "height": 1,
            "font": (CONFIG["FONT_FAMILY"], CONFIG["BUTTON_FONT_SIZE"]),
            "relief": "raised",
            "borderwidth": 2,
        }

        # 同意按钮
        tk.Button(
            frame,
            text="当然好呀 ❤",
            command=self._on_accept,
            bg=CONFIG["PRIMARY_COLOR"],
            fg="white",
            **button_style,
        ).pack(side=tk.LEFT, padx=10)

        # 考虑按钮
        tk.Button(
            frame,
            text="再想想 💭",
            command=self._on_reject,
            bg="#FFB6C1",
            fg="white",
            **button_style,
        ).pack(side=tk.LEFT, padx=10)

    def _create_heart_window(self) -> None:
        """创建心形动画窗口"""
        heart_window = tk.Toplevel(self.root)
        heart_window.attributes("-fullscreen", True)  # 全屏显示
        heart_window.title("❤ Love You Forever ❤")

        # 创建画布
        canvas = tk.Canvas(
            heart_window,
            width=CONFIG["WINDOW_WIDTH"],
            height=CONFIG["WINDOW_HEIGHT"],
            bg=CONFIG["BACKGROUND_COLOR"],
            highlightthickness=0,
        )
        canvas.pack(expand=True)

        # 创建文字标签
        self._create_love_message(heart_window)

        # 启动动画
        self.heart_animation = HeartAnimation()
        self._animate(heart_window, canvas)

        # 设置关闭事件
        heart_window.protocol("WM_DELETE_WINDOW", self.root.quit)

        # 绑定ESC键退出全屏
        heart_window.bind(
            "<Escape>", lambda e: heart_window.attributes("-fullscreen", False)
        )

    def _create_love_message(self, window: tk.Tk) -> None:
        """创建爱心消息"""
        message = tk.Label(
            window,
            text="I Love You Forever!",
            bg=CONFIG["BACKGROUND_COLOR"],
            fg=CONFIG["SECONDARY_COLOR"],
            font=(CONFIG["FONT_FAMILY"], 30, "bold"),
        )
        message.place(relx=0.5, rely=0.5, anchor=CENTER)

        # 添加消息切换动画
        self._animate_message(message)

    def _animate_message(self, label: tk.Label) -> None:
        """动画显示消息"""

        def update_message():
            label.configure(text=random.choice(self.love_messages))
            label.after(3000, update_message)  # 每3秒更换一次消息

        update_message()

    def _animate(self, window: tk.Tk, canvas: tk.Canvas) -> None:
        """动画循环"""
        self.heart_animation.render(canvas)
        window.after(CONFIG["ANIMATION_SPEED"], self._animate, window, canvas)

    def _on_accept(self) -> None:
        """接受按钮回调"""
        self.root.withdraw()  # 隐藏主窗口
        messagebox.showinfo("❤ Sweet Love ❤", "太好了！我们的故事正式开始了！")
        self._create_heart_window()

    def _on_reject(self) -> None:
        """拒绝按钮回调"""
        messages = [
            "让我们开始一段浪漫的故事吧❤",
            "你就答应我吧，好不好？🌹",
            "我会一直等待你的答案💕",
            "这次真的不考虑一下吗？✨",
            "你是我最特别的人，请给我一次机会💝",
        ]

        # 获取按钮的父框架
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame):
                # 遍历框架中的按钮
                for button in widget.winfo_children():
                    if (
                        isinstance(button, tk.Button)
                        and button.cget("text") == "再想想 💭"
                    ):
                        # 移动按钮位置的随机范围
                        x = random.randint(0, CONFIG["DIALOG_WIDTH"] - 100)
                        y = random.randint(0, CONFIG["DIALOG_HEIGHT"] - 50)
                        button.place(x=x, y=y)
                        break

        # 显示随机消息
        messagebox.showinfo("❤ Sweet Message ❤", random.choice(messages))

        # 增加关闭尝试计数
        self.close_attempts += 1
        if self.close_attempts >= CONFIG["MAX_CLOSE_ATTEMPTS"]:
            messagebox.showinfo("❤ Love Decision ❤", "看来这是命中注定的缘分呢！")
            self._on_accept()

    def run(self) -> None:
        """运行应用程序"""
        self.root.mainloop()


if __name__ == "__main__":
    app = HeartApp()
    app.run()

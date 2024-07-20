import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


# 加载哆啦A梦图片并提取所有轮廓点
def extract_contour_points(image_path):
    print(f"Loading image from {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(gray, 100, 200)

    # 使用形态学操作进行膨胀和腐蚀
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 合并所有轮廓点
    points = []
    for contour in contours:
        contour_points = contour[:, 0, 0] + 1j * contour[:, 0, 1]
        points.extend(contour_points)

    points = np.array(points)
    print(f"Extracted {len(points)} contour points")
    return points


# 计算傅里叶系数
def calculate_fourier_coefficients(points, N):
    c = np.fft.fft(points) / N
    print(f"Calculated {len(c)} Fourier coefficients")
    return c


# 使用傅里叶级数重建轮廓
def reconstruct_curve(c, N, num_points=1000, k_max=None):
    t = np.linspace(0, 2 * np.pi, num_points)
    curve = np.zeros(num_points, dtype=complex)
    if k_max is None:
        k_max = N // 2
    for k in range(-k_max, k_max):
        curve += c[k] * np.exp(1j * k * t)
    return curve


# 动态绘制
def animate_doraemon(image_path, num_frames=200):
    points = extract_contour_points(image_path)
    N = len(points)
    c = calculate_fourier_coefficients(points, N)

    fig, ax = plt.subplots(figsize=(8, 8))
    (line,) = ax.plot([], [], lw=2)
    ax.axis("equal")
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_title("Doraemon using Fourier Series")

    def init():
        line.set_data([], [])
        return (line,)

    def update(frame):
        k_max = 1 + frame * (N // 2) // num_frames
        reconstructed_curve = reconstruct_curve(c, N, k_max=k_max)
        line.set_data(reconstructed_curve.real, reconstructed_curve.imag)
        return (line,)

    anim = FuncAnimation(
        fig, update, frames=num_frames, init_func=init, blit=True, repeat=False
    )
    plt.show()

    return anim


# 主程序
if __name__ == "__main__":
    image_path = "doraemon.png"

    # 检查图像加载是否正确
    try:
        image = cv2.imread(image_path)
        if image is not None:
            print(f"Successfully loaded image: {image_path}")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Loaded Image")
            plt.axis("off")
            plt.show()
        else:
            print(f"Failed to load image: {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")

    # 提取轮廓点
    try:
        points = extract_contour_points(image_path)
        if len(points) > 0:
            print("Successfully extracted contour points")
            plt.plot(points.real, points.imag, "o", markersize=1)
            plt.title("Extracted Contour Points")
            plt.axis("equal")
            plt.show()
        else:
            print("No contour points extracted")
    except Exception as e:
        print(f"Error extracting contour points: {e}")

    # 动态绘制哆啦A梦
    try:
        anim = animate_doraemon(image_path)
    except Exception as e:
        print(f"Error in animation: {e}")

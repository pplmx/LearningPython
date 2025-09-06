import cv2
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, fixed, interact


def extract_and_display_contours(image_path, canny_threshold1, canny_threshold2, kernel_size, iterations):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Morphological Operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    eroded = cv2.erode(dilated, kernel, iterations=iterations)

    # Find Contours
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Plot Original Image and Contours
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    contour_image = np.zeros_like(image)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)
    axes[1].imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Contours (Canny: [{canny_threshold1}, {canny_threshold2}], Kernel: {kernel_size}, Iterations: {iterations})"
    )
    axes[1].axis("off")

    plt.show()

    return contours


# Interact function
def interactive_contours(image_path):
    interact(
        extract_and_display_contours,
        image_path=fixed(image_path),
        canny_threshold1=IntSlider(min=0, max=255, step=1, value=100, description="Canny Threshold 1"),
        canny_threshold2=IntSlider(min=0, max=255, step=1, value=200, description="Canny Threshold 2"),
        kernel_size=IntSlider(min=1, max=10, step=1, value=3, description="Kernel Size"),
        iterations=IntSlider(min=1, max=10, step=1, value=1, description="Iterations"),
    )


# Run the interactive widget
image_path = "doraemon.png"
interactive_contours(image_path)

"""Load an image, detect an edge and do some filtering to produce a cartoon version of the original"""
import cv2
import numpy as np

line_size = 7
total_color = 9
blur_value = 5
filename = "./images/photo.png"


def read_file(filename: str) -> np.ndarray:
    img = cv2.imread(filename)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    return img


def edge_mask(img: np.ndarray, line_size: int, blur_value: int) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        line_size,
        blur_value,
    )
    return edges


def color_quantization(img: np.ndarray, k: int) -> np.ndarray:
    """Color Quantization, Reduce the amount of colors in the input image using Kmeans"""
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(
        data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


def main() -> None:
    img = read_file(filename)
    edges = edge_mask(img, line_size, blur_value)
    img = color_quantization(img, total_color)
    blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    cv2.imshow("image", cartoon)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()

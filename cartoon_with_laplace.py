import cv2
import numpy as np
from utils.load_images_folder import load_images_from_folder
from utils.load_2_images import display_images

line_size = 7
total_color = 6
blur_value = 5
images_folder = "./images"


def read_file(filename):
    img = cv2.imread(filename)
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    return img


def resize(img, preserve: int = 0, selected_width=800, selected_height=600):
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    if preserve == 1:
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    else:
        width = selected_width
        height = selected_height
        dim = (width, height)
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print(resized_img.shape)
    return resized_img


def edge_mask(img, line_size, blur_value):
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


def filtering(img):
    # Apply some Gaussian blur on the image
    img_gb = cv2.GaussianBlur(img, (7, 7), 0)
    # Apply some Median blur on the image
    img_mb = cv2.medianBlur(img_gb, 5)
    # Apply a bilateral filer on the image
    img_bf = cv2.bilateralFilter(img_mb, 5, 80, 80)
    images = [img, img_gb, img_mb, img_bf]
    return images


def edge_detection(img, img_gb, img_mb, img_bf):
    """Use the laplace edge detector"""
    img_lp_im = cv2.Laplacian(img, cv2.CV_8U, ksize=5)
    img_lp_gb = cv2.Laplacian(img_gb, cv2.CV_8U, ksize=5)
    img_lp_mb = cv2.Laplacian(img_mb, cv2.CV_8U, ksize=5)
    img_lp_al = cv2.Laplacian(img_bf, cv2.CV_8U, ksize=5)
    edged_images = [img_lp_im, img_lp_gb, img_lp_mb, img_lp_al]
    return edged_images


def convert_greyscale(img_lp_im, img_lp_gb, img_lp_mb, img_lp_al):
    img_lp_im_grey = cv2.cvtColor(img_lp_im, cv2.COLOR_BGR2GRAY)
    img_lp_gb_grey = cv2.cvtColor(img_lp_gb, cv2.COLOR_BGR2GRAY)
    img_lp_mb_grey = cv2.cvtColor(img_lp_mb, cv2.COLOR_BGR2GRAY)
    img_lp_al_grey = cv2.cvtColor(img_lp_al, cv2.COLOR_BGR2GRAY)
    greyscale_images = [img_lp_im_grey, img_lp_gb_grey, img_lp_mb_grey, img_lp_al_grey]
    return greyscale_images


def thresholding(img_lp_im_grey, img_lp_gb_grey, img_lp_mb_grey, img_lp_al_grey):
    """Remove noise and use Otsu thresholding"""
    # Remove some additional noise
    blur_im = cv2.GaussianBlur(img_lp_im_grey, (5, 5), 0)
    blur_gb = cv2.GaussianBlur(img_lp_gb_grey, (5, 5), 0)
    blur_mb = cv2.GaussianBlur(img_lp_mb_grey, (5, 5), 0)
    blur_al = cv2.GaussianBlur(img_lp_al_grey, (5, 5), 0)
    # Apply a threshold (Otsu)
    _, tresh_im = cv2.threshold(blur_im, 245, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, tresh_gb = cv2.threshold(blur_gb, 245, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, tresh_mb = cv2.threshold(blur_mb, 245, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, tresh_al = cv2.threshold(blur_al, 245, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold_images = [tresh_im, tresh_gb, tresh_mb, tresh_al]
    return threshold_images


def invert_blackwhite(tresh_im, tresh_gb, tresh_mb, tresh_al):
    inverted_original = cv2.subtract(255, tresh_im)
    inverted_GaussianBlur = cv2.subtract(255, tresh_gb)
    inverted_MedianBlur = cv2.subtract(255, tresh_mb)
    inverted_Bilateral = cv2.subtract(255, tresh_al)
    inverted_images = [
        inverted_original,
        inverted_GaussianBlur,
        inverted_MedianBlur,
        inverted_Bilateral,
    ]
    return inverted_images


def color_quantization(img, k):
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


# images = load_images_from_folder(images_folder)
img = read_file("./images/photo.png")
filtered_images = filtering(img)
edge_images = edge_detection(
    filtered_images[0], filtered_images[1], filtered_images[2], filtered_images[3]
)
display_images(edge_images)
greyscale_images = convert_greyscale(
    edge_images[0], edge_images[1], edge_images[2], edge_images[3]
)
threshold_images = thresholding(
    greyscale_images[0], greyscale_images[1], greyscale_images[2], greyscale_images[3]
)
inverted_images = invert_blackwhite(
    threshold_images[0], threshold_images[1], threshold_images[2], threshold_images[3]
)
display_images(inverted_images)
less_colors_image = color_quantization(img, total_color)

inverted_bilateral = inverted_images[3]
# Convert the mask image back to color
inverted_Bilateral = cv2.cvtColor(inverted_bilateral, cv2.COLOR_GRAY2RGB)
# Combine the edge image and the binned image
cartoon_Bilateral = cv2.bitwise_and(inverted_Bilateral, less_colors_image)
# Save the image
cv2.imwrite("./images/result/CartoonImage.png", cartoon_Bilateral)


# edges = edge_mask(img, line_size, blur_value)
# img = color_quantization(img, total_color)
# blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
# cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
# cv2.imshow("image", cartoon)
# cv2.waitKey(0)

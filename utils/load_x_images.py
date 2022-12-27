""" Loads and displays images in a grid using the glob module and matplotlib"""
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

file = "./images/*.png"
glob.glob(file)
# Using List Comprehension to read all images
images = [cv2.imread(image) for image in glob.glob(file)]


def display_images(images: "list[np.ndarray[int]]") -> None:
    """Display x amount of images from a list"""
    # Define a figure of size (8, 8)
    fig = plt.figure(figsize=(8, 8))
    # Define row and cols in the figure
    rows, cols = 2, 2
    # Display first images
    for j in range(0, cols * rows):
        fig.add_subplot(rows, cols, j + 1)
        plt.imshow(images[j])
    plt.show()

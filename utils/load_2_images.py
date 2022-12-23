import glob
import cv2
import matplotlib.pyplot as plt

file = "./images/*.png"
glob.glob(file)
# Using List Comprehension to read all images
images = [cv2.imread(image) for image in glob.glob(file)]


def display_images(images):
    # Define a figure of size (8, 8)
    fig = plt.figure(figsize=(8, 8))
    # Define row and cols in the figure
    rows, cols = 2, 2
    # Display first four images
    for j in range(0, cols * rows):
        fig.add_subplot(rows, cols, j + 1)
        plt.imshow(images[j])
    plt.show()

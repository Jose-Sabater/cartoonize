import cv2
import os
import numpy as np


def load_images_from_folder(folder: str) -> "list[np.ndarray]":
    """Loads all images in a folder"""
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

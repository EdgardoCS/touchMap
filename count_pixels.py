"""
author: EdgardoCS @FSU Jena
date: 27/03/2025
"""
import cv2
from PIL import Image
import numpy as np

image = Image.open("Body chart - women-painted.png")

pixels = list(image.getdata())

w,h = image.size

n_pixels = image.load()
black_pixels = 0
white_pixels = 0

for i in range(image.size[0]):  # for every col:
    for j in range(image.size[1]):  # for every row:
        if n_pixels[i,j][0] == 0:
            black_pixels += 1
        else:
            white_pixels += 1

print(black_pixels)
print(white_pixels)
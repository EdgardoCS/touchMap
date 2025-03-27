"""
author: EdgardoCS @FSU Jena
date: 25/03/2025
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as time
from PIL import Image,ImageChops
from nbformat.v4 import new_raw_cell


def rgba2rgb(RGBA_color):
    """
    background color is white (255.255.255)
    :param RGBA_color:
    :return:
    """
    input_pixel = RGBA_color
    alpha_value = input_pixel[3]/255
    new_r = int((1-alpha_value) * 255+ alpha_value * input_pixel[0])
    new_b = int((1-alpha_value) * 255+ alpha_value * input_pixel[1])
    new_g = int((1-alpha_value) * 255+ alpha_value * input_pixel[2])
    new_pixel = (new_r, new_b, new_g)

    return new_pixel


# Load image
image = Image.open("sexWomen.png")

# Get height and width
w, h = image.size

# Rewrite pixels to eliminate A band (transparency)
new_pixels = image.load()
for i in range(image.size[0]):    # for every col:
    for j in range(image.size[1]):
        new_pixels[i,j] = rgba2rgb(new_pixels[i,j])
#image.save('sexWomen_bg.png', "png")
#image.show()

new_image = cv.imread("sexWomen_bg.png")
# Divide image in half
half = w//2
front = new_image[:, :half]
back = new_image[:, half:]

images = [new_image, front, back]
titles = ["original", "front", "back"]


for i in range(3):
    plt.subplot(1, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
#plt.show()
cv.imwrite('front.png', front)
cv.imwrite('back.png', back)

# Read image, front and back using PIL.Image
im1 = Image.open('front.png')
im2 = Image.open('back.png')

im1 = im1.convert('1')
im2 = im2.convert('1')

diff = ImageChops.logical_and(im1, im2)
if diff.getbbox():
    diff.show()
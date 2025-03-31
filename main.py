"""
author: EdgardoCS @FSU Jena
date: 25/03/2025
"""
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as time
from PIL import Image, ImageChops
from nbformat.v4 import new_raw_cell


def rgba2rgb(RGBA_color):
    """
    background color is white (255.255.255)
    :param RGBA_color:
    :return:
    """
    input_pixel = RGBA_color
    alpha_value = input_pixel[3] / 255
    new_r = int((1 - alpha_value) * 255 + alpha_value * input_pixel[0])
    new_b = int((1 - alpha_value) * 255 + alpha_value * input_pixel[1])
    new_g = int((1 - alpha_value) * 255 + alpha_value * input_pixel[2])
    new_pixel = (new_r, new_b, new_g)
    return new_pixel


def intensityNormalization(pixel_data, maximum):
    """

    :param pixel_data:
    :param maximum:
    :return:
    """
    input_pixel = pixel_data
    alpha_value = input_pixel[3] / 255
    new_r = int(input_pixel[0])
    new_b = int(input_pixel[1])
    new_g = int(input_pixel[2])
    new_a = int((alpha_value * 100 / maximum))
    new_pixel = (new_r, new_b, new_g, new_a)
    return new_pixel


def calculateHighest(pixel_data, width, height):
    """
    [0]Red; [1]Blue; [2]Green; [3]Transparency
    :param pixel_data:
    :param width:
    :param height:
    :return: maximum value of transparency from A band, for this case this should be the 100% of intensity of touch
    """
    total = []
    input_pixel = pixel_data
    for i in range(int(width / 2)):  # for every col:
        for j in range(int(height / 2)):  # for every row:
            alpha_value = input_pixel[i, j][3] / 255  # read data from the A band
            total.append(alpha_value)
    maximum = max(total)
    return maximum


location1 = "data/LovePreferredWomen/"
location2 = "data/LoveReceivedWomen/"
location = location2

target1 = "w1"
target2 = "w2"
target = target1

# Load images
woman1 = Image.open(location + "w1" + ".png")
woman2 = Image.open(location + "w2" + ".png")

# Get dimensions
w1, h1 = woman1.size  # should be 800 x 300
w2, h2 = woman2.size

# ROI
box = (0, 0, 400, 300)
woman1_front = woman1.crop(box)
woman2_front = woman2.crop(box)

# Read pixel information for each subject
w1_pixels = woman1_front.load()
w2_pixels = woman2_front.load()

"""
woman1_front.save(location + "w1" + "_bg.png", "png")
woman2_front.save(location + "w2" + "_bg.png", "png")
new1 = Image.open(location + "w1" + "_bg.png")
new2 = Image.open(location + "w2" + "_bg.png")
pixel1 = list(new1.getdata())
pixel2 = list(new2.getdata())

df1 = pd.DataFrame(pixel1)
df2 = pd.DataFrame(pixel2)
df1.to_csv("woman1_front.csv")
df2.to_csv("woman2_front.csv")
"""

# Calculates the highest intensity of color, to be used as the 100% (to normalize the data)
highest1 = calculateHighest(w1_pixels, w1, h1)
highest2 = calculateHighest(w2_pixels, w1, h1)

output_pixel1 = woman1_front.load()
output_pixel2 = woman2_front.load()

# Now that we have the highest possible number of intensity, lets normalize our data
for i in range(woman1_front.size[0]):  # for every col:
    for j in range(woman1_front.size[1]):  # for every row:
        output_pixel1[i, j] = intensityNormalization(w1_pixels[i, j], highest1)  # Image 1
        output_pixel2[i, j] = intensityNormalization(w2_pixels[i, j], highest2)  # Image 2

"""
woman1_front.save(location + "w1" + "_bg_a.png", "png")
woman2_front.save(location + "w2" + "_bg_a.png", "png")
new1 = Image.open(location + "w1" + "_bg_a.png")
new2 = Image.open(location + "w2" + "_bg_a.png")
pixel1 = list(new1.getdata())
pixel2 = list(new2.getdata())

df1 = pd.DataFrame(pixel1)
df2 = pd.DataFrame(pixel2)
df1.to_csv("woman1_front_a.csv")
df2.to_csv("woman2_front_a.csv")
"""

new_image = Image.new(mode="RGBA", size=(400, 300))
merge_pixels = new_image.load()
for i in range(woman1_front.size[0]):
    for j in range(woman1_front.size[1]):
        if output_pixel1[i, j][0] == 255 or output_pixel2[i, j][0] == 255:
            new_r = 255
            new_b = 0
            new_g = 0
            new_alpha = output_pixel1[i, j][3] + output_pixel2[i, j][3]
            new_pixel = (new_r, new_b, new_g, new_alpha)
            merge_pixels[i, j] = new_pixel
        else:
            new_r = 0
            new_b = 0
            new_g = 0
            new_alpha = 0
            new_pixel = (new_r, new_b, new_g, new_alpha)
            merge_pixels[i, j] = new_pixel
        #print(merge_pixels[i, j])

#new_image.save(location + "mergeData" + ".png", "png")
# temp = Image.open(location +"mergeData.png")
# pixel = list(temp.getdata())
# df = pd.DataFrame(pixel)
# df.to_csv("mergeData.csv")


# # Rewrite pixels to eliminate A band (transparency)
# new_pixels = image.load()
# for i in range(image.size[0]):  # for every col:
#     for j in range(image.size[1]):  # for every row:
#         new_pixels[i,j] = rgba2rgb(new_pixels[i,j])
# image.save(location + target + "_bg.png", "png")
# #image.show()
#
#
# # Read image, front and back using PIL.Image
# im1 = Image.open('front.png')
# im2 = Image.open('back.png')
#
# im1 = im1.convert('1')
# im2 = im2.convert('1')
#
# diff = ImageChops.logical_and(im1, im2)
# if diff.getbbox():
#     diff.show()

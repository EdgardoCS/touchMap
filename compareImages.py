"""
author: EdgardoCS @FSU Jena
date: 28/03/2025
"""

import numpy as np
from PIL import Image
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
from mayavi import mlab


def plotHeatmap(image_map1, image_map2, outline):
    """

    :param image_map1:
    :param image_map2:
    :param outline:
    :return:
    """
    plt.imshow(image_map1, cmap='hot', interpolation='nearest')
    plt.imshow(outline, alpha=0.5)
    plt.show()

    # f, axs = plt.subplots(1, 2)
    # axs[0].imshow(image_map1, cmap='hot', interpolation='nearest')
    # axs[1].imshow(image_map2, cmap='hot', interpolation='nearest')
    # axs[0].axis("off")
    # axs[1].axis("off")
    # plt.subplots_adjust(wspace=0.0)
    # plt.show()


def plotSurface(image_map, x_axis, y_axis):
    """

    :param image_map:
    :param x_axis:
    :param y_axis:
    :return:
    """
    z = image_map  # Use the values directly as height
    # z = (image_map - np.min(image_map)) / (np.max(image_map) - np.min(image_map))  # Normalize to [0,1]
    # z = (image_map - np.min(image_map)) / (np.max(image_map) - np.min(image_map))  # Normalize
    # z = np.clip(z)  # Ensure values stay between 0 and 1

    # t value from https://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf

    new_z = np.where(z < 12.71, np.nan, z)
    nx = x_axis
    ny = y_axis
    x1 = np.linspace(0, 10, ny)
    y1 = np.linspace(0, 10, nx)
    x, y = np.meshgrid(x1, y1)

    # print(np.count_nonzero(z), "original z")
    # print(np.count_nonzero(new_z), "new z")

    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')
    ax.view_init(elev=0, azim=-50, roll=0)
    ax.plot_wireframe(x, y, new_z, rstride=10, cstride=10)
    plt.title("t values above p (p = 0,05, t = 12,71)")

    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')
    ax.view_init(elev=0, azim=-50, roll=0)
    ax.plot_wireframe(x, y, z, rstride=10, cstride=10)
    plt.title("all t values")
    plt.show()


def calculate_t(input1, input2):
    """

    :param input1:
    :param input2:
    :return:
    """
    u = 0
    t = (np.mean([input1, input2]) - u) / (stats.sem([input1, input2]))

    return t


def count_match(input1, input2):
    """

    :param input1:
    :param input2:
    :return:
    """
    n_match = 0
    n_unmatch = 0

    for i in range(0, input1.shape[0]):
        for j in range(0, input1.shape[1]):
            if input1[i, j][3] != 0 and input2[i, j][3] != 0:
                n_match += 1
            else:
                n_unmatch += 1
    total_match = (n_match * 100) / (input1.shape[0] * input1.shape[1])
    return total_match


# TODO: calculate % of match between two images (done)
# Fit the image on body chart (heatmap)
# TODO: FDR

# Load images
image1 = Image.open('data/Love Preferred Women/mergeData.png')  # could use .convert("RGBA")
image2 = Image.open('data/Love Received Women/mergeData.png')

# We want to compare front against front, and back against back
front1 = image1.crop((0, 0, 400, 300))
back1 = image1.crop((400, 0, 800, 300))

front2 = image2.crop((0, 0, 400, 300))
back2 = image2.crop((400, 0, 800, 300))

# Convert to numpy arrays
arr1 = np.array(front1)
print(arr1.shape)
arr2 = np.array(front2)
arr3 = np.array(back1)
arr4 = np.array(back2)

match_percentage = count_match(arr1, arr2)
print("% of match pixels between w1 and w2 is", f' {match_percentage:}%')

# Extract alpha chanel (transparency)
alpha1 = arr1[:, :, 3].astype(float)
alpha2 = arr2[:, :, 3].astype(float)
alpha3 = arr3[:, :, 3].astype(float)
alpha4 = arr4[:, :, 3].astype(float)

# Compute absolute differences (not sure about this part)
front_map = np.abs(alpha1 - alpha2)  # This acts as a "difference significance" map
back_map = np.abs(alpha3 - alpha4)

# Load body map outline
body_outline = Image.open("data/Female_bodychart_nobg.png")

t_map = np.zeros(shape=alpha1.shape)
for i in range(front_map.shape[0]):
    for j in range(front_map.shape[1]):
        t_map[i, j] = calculate_t(alpha1[i, j], alpha2[i, j])

# plotHeatmap(front_map, back_map, body_outline)
# plotSurface(t_map, 300, 400)

"""
# Plot the results
plt.imshow(final_im ,cmap='Greys', interpolation='nearest')
plt.colorbar(label="Touch intensity")

plt.imshow(body_outline, alpha=0.1)
plt.title("Pixel-wise Heatmap")
plt.show()

new = []
for i in range(diff_map1.shape[0]):
    for j in range(diff_map1.shape[1]):
        if diff_map1[i, j] != 0:
            new.append([i, j, diff_map1[i, j]])
        else:
            new.append([i, j, 0])

print(new)
new = np.asarray(new)


plt.imshow(new)
plt.show()

#x = new[1, :, :].astype(float)
#y = new[:, 2, :].astype(float)
#z = new[:, :, 3].astype(float)

"""

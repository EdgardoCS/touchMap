"""
author: EdgardoCS @FSU Jena
date: 28/03/2025
"""

import math
import numpy as np
from PIL import Image
from matplotlib import cm
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests


def plotHeatmap(image_map1, image_map2, outline):
    """

    :param image_map1:
    :param image_map2:
    :param outline:
    :return:
    """
    #target_w = 460
    #target_h = 800
    from scipy.ndimage import zoom


    front_and_back = np.hstack((image_map1, image_map2))

    print(front_and_back.shape)
    print(outline.size)

    resized_arr = zoom(front_and_back, (800/300, 400/800), order=1)

    plt.imshow(resized_arr, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Touch intensity")
    plt.imshow(outline)

    plt.show()

    # f, axs = plt.subplots(1, 2)
    # axs[0].imshow(image_map1, cmap='hot', interpolation='nearest')
    # axs[1].imshow(image_map2, cmap='hot', interpolation='nearest')
    # axs[0].axis("off")
    # axs[1].axis("off")
    # plt.subplots_adjust(wspace=0.0)
    # plt.show()


def plotSurface(image_map, pvalues_map, x_axis, y_axis):
    """

    :param image_map:
    :param x_axis:
    :param y_axis:
    :return:
    """
    z = image_map  # Use the values directly as height

    # t value from https://www.sjsu.edu/faculty/gerstman/StatPrimer/t-table.pdf

    z_significant = np.where(z < 12.71, np.nan, z)
    nx = x_axis
    ny = y_axis
    x1 = np.linspace(0, 10, ny)
    y1 = np.linspace(0, 10, nx)
    x, y = np.meshgrid(x1, y1)

    nan_mask = np.isnan(pvalues_map)
    valid_p_values = pvalues_map[~nan_mask]

    rejected, pvals_corrected_valid, _, _ = multipletests(valid_p_values, alpha=0.05, method='fdr_bh')
    pvals_corrected = np.full_like(pvalues_map, np.nan, dtype=np.float64)
    pvals_corrected[~nan_mask] = pvals_corrected_valid

    # f, axs = plt.subplots(2, 2)
    # axs[0,0].imshow(z, cmap='hot', interpolation='nearest')
    # axs[0,1].imshow(pvalues_map, cmap='hot', interpolation='nearest')
    # axs[1,1].imshow(pvals_corrected, cmap='hot', interpolation='nearest')
    # plt.show()

    plt.imshow(z, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Touch intensity")

    plt.imshow(pvalues_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Touch intensity")

    plt.imshow(pvals_corrected, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Touch intensity")

    plt.show()

    # fig1 = plt.figure()
    # ax = fig1.add_subplot(projection='3d')
    # ax.view_init(elev=0, azim=-50, roll=0)
    # ax.plot_wireframe(x, y, z_significant, rstride=10, cstride=10)
    # plt.title("significat t values (p = 0,05, t = 12,71)")

    # fig3 = plt.figure()
    # ax = fig3.add_subplot(projection='3d')
    # ax.view_init(elev=0, azim=-50, roll=0)
    # ax.plot_wireframe(x, y, pvals_corrected, rstride=10, cstride=10)
    # plt.title("adjusted p-values")
    #
    # plt.show()


def calculate_t(input1, input2):
    """

    :param input1:
    :param input2:
    :return:
    """
    ttest = stats.ttest_1samp([input1, input2], popmean=0.0)
    t = ttest.statistic
    p = ttest.pvalue
    # u = 0
    # if input1 == input2 and input1 != 0 and input2 != 0: # just to avoid inf
    #     input2 = input2 + 0.1
    #     t = (np.mean([input1, input2]) - u) / (stats.sem([input1, input2]))
    # else:
    #     t = (np.mean([input1, input2]) - u) / (stats.sem([input1, input2])) # this is the actual code that works
    return t, p


def count_match(input1, input2):
    """

    :param input1:
    :param input2:
    :return:
    """
    n_match = 0
    n_unmatch = 0
    ### WRONG! it is not about how many matched pixels are among the whole, it is only among the one that have signal!!
    ### RECALCULATE
    for i in range(0, input1.shape[0]):
        for j in range(0, input1.shape[1]):
            if input1[i, j][3] != 0 and input2[i, j][3] != 0:
                n_match += 1
            else:
                n_unmatch += 1
    total_match = (n_match * 100) / (input1.shape[0] * input1.shape[1])
    return total_match


# TODO:
#  calculate % of match between two images (done)
#  Fit the image on body chart (heatmap)
#  FDR

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
p_map = np.zeros(shape=alpha1.shape)
for i in range(front_map.shape[0]):
     for j in range(front_map.shape[1]):
         t_map[i, j], p_map[i, j] = calculate_t(alpha1[i, j], alpha2[i, j])

#plotHeatmap(front_map, back_map, body_outline)
plotSurface(t_map, p_map, 300, 400)
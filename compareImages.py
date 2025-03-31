"""
author: EdgardoCS @FSU Jena
date: 28/03/2025
"""

import numpy as np
from PIL import Image
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load images
image1 = Image.open('data/preferred/mergeData0.png') # could use .convert("RGBA")
image2 = Image.open('data/received/mergeData0.png')

# Convert to numpy arrays
arr1 = np.array(image1)
arr2 = np.array(image2)

# Extract alpha chanel (transparent)
alpha1 = arr1[:, :, 3].astype(float)
alpha2 = arr2[:, :, 3].astype(float)

# Compute absolute differences
diff_map = np.abs(alpha1 - alpha2)  # This acts as a "difference significance" map

outline = Image.open("Body_chart_-_women-removebg-preview.png")
outline_resized = outline.resize((800, 300))
plt.subplot(131)
plt.imshow(outline_resized)
# plt.show()

box = (0, 0, 400, 300)
plt.subplot(132)
outline_cropped = outline_resized.crop(box)
plt.imshow(outline_cropped)

# Plot the results
plt.subplot(133)
plt.imshow(diff_map, cmap='hot', interpolation='nearest')
plt.colorbar(label="Touch intensity")

plt.imshow(outline_cropped, alpha=0.7)
plt.title("Pixel-wise Heatmap")
plt.show()
"""
author: EdgardoCS @FSU Jena
date: 28/03/2025
"""

import numpy as np
from PIL import Image
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm

# Load images
image1 = Image.open('data/Love Preferred Women/mergeData.png')  # could use .convert("RGBA")
image2 = Image.open('data/Love Received Women/mergeData.png')

front1 = image1.crop((0, 0, 400, 300))
back1 = image1.crop((400, 0, 800, 300))

front2 = image2.crop((0, 0, 400, 300))
back2 = image2.crop((400, 0, 800, 300))

# Convert to numpy arrays
arr1 = np.array(front1)
arr2 = np.array(front2)
arr3 = np.array(back1)
arr4 = np.array(back2)

# Extract alpha chanel (transparent)
alpha1 = arr1[:, :, 3].astype(float)
alpha2 = arr2[:, :, 3].astype(float)
alpha3 = arr3[:, :, 3].astype(float)
alpha4 = arr4[:, :, 3].astype(float)

# Compute absolute differences
diff_map1 = np.abs(alpha1 - alpha2)  # This acts as a "difference significance" map
diff_map2 = np.abs(alpha3 - alpha4)

body_outline = Image.open("data/Female_bodychart.png")

# f, axs = plt.subplots(1, 2)
# axs[0].imshow(diff_map1, cmap='hot', interpolation='nearest')
# axs[1].imshow(diff_map2, cmap='hot', interpolation='nearest')
# axs[0].axis("off")
# axs[1].axis("off")
# plt.subplots_adjust(wspace=0.0)
# plt.show()
Z = diff_map1  # Use the values directly as height
Z = (diff_map1 - np.min(diff_map1)) / (np.max(diff_map1) - np.min(diff_map1))  # Normalize to [0,1]
Z = (diff_map1 - np.min(diff_map1)) / (np.max(diff_map1) - np.min(diff_map1))  # Normalize
Z = np.clip(Z, 0, 1)  # Ensure values stay between 0 and 1

nx, ny = 300, 400
data = np.zeros((nx, ny))
x = np.linspace(0, 10, ny)
y = np.linspace(0, 10, nx)
X, Y = np.meshgrid(x, y)

# Create the figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, facecolors=cm.viridis(data),
                        rstride=1, cstride=1, alpha=0.7,
                        edgecolor='none')

# Add color bar
m = cm.ScalarMappable(cmap=cm.viridis)
m.set_array(data)
fig.colorbar(m, ax=ax, shrink=0.5, aspect=10)

# Labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Transparency as Z')

plt.show()

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
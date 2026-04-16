# -*- coding: utf-8 -*-
"""
Tests the cropping and mask functionality of pybundle

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import context

import pybundle

filterSize = 2.5

# Load test images
img = np.array(Image.open("data/usaf1.tif"))
calibImg = np.array(Image.open("data/usaf1_background.tif"))


# Locate the bundle
loc = pybundle.find_bundle(calibImg)


# Generate a mask by specifying the location of the bundle
mask1 = pybundle.get_mask(calibImg, loc)


# Generate a mask without specifying the location of the bundle and apply it to image
imgProc1 = pybundle.auto_mask(img)


# Apply mask1 to an imqge
imgProc2 = pybundle.apply_mask(img, mask1)


# Generate a mask and then crop image around mask
imgProc3 = pybundle.auto_mask_crop(img)[0]


plt.figure(dpi=300)
plt.imshow(mask1, cmap='gray')
plt.title('Mask from get_mask')

plt.figure(dpi=300)
plt.imshow(imgProc1, cmap='gray')
plt.title('Image auto_mask')

plt.figure(dpi=300)
plt.imshow(imgProc2, cmap='gray')
plt.title('Image masked')

plt.figure(dpi=300)
plt.imshow(imgProc3, cmap='gray')
plt.title('Image auto_mask_crop')
# -*- coding: utf-8 -*-
"""
Example of removal of fibre bundle core pattern from colour images by 
Delaunay triangulation and triangular linear interpolation.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time

from PIL import Image

import context    # Add relative path to get PyBundle

import os
import pybundle
from pybundle import PyBundle

# We load in two images, an image with uniform illumination for calibation
# and a image of letters to demonstrate core removal
img = np.array(Image.open(r"..\test\data\bundle_colour_1.tif"))
calibImg = np.array(Image.open(r"..\test\data\bundle_colour_1_background.tif"))


# Parameters for reconstruction
coreSize = 3          # Estimated core size used when searching for cores
gridSize = 512        # Number of pixels in reconstructed image
filterSize = None     # Pre-Gaussian filter sigma


pyb = pybundle.PyBundle(coreMethod = PyBundle.TRILIN, 
                        calibImage = calibImg, 
                        normaliseImage = calibImg, 
                        coreSize = 3, 
                        gridSize = 512)

# One-time calibration
t1 = time.perf_counter()
pyb.calibrate()
t2 = time.perf_counter()
print(f"Calibration took: {round((t2-t1) * 1000)} ms")

# Image reconstruction without Numba
imgRecon = pyb.process(img)
t1 = time.perf_counter()
imgRecon = pyb.process(img)
t2 = time.perf_counter()
print(f"Colour linear interpolation took: {round((t2-t1) * 1000)} ms")

# Display reconstructed image
plt.figure(dpi = 150)
plt.imshow(img, cmap='gray')
plt.title('Raw Image')

# Display reconstructed image
plt.figure(dpi = 150)
plt.imshow(imgRecon / 1024, cmap='gray')
plt.title('Reconstruction by interpolation')

plt.show()


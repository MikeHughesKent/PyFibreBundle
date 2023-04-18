# -*- coding: utf-8 -*-
"""
Test removal of fibre bundle core pattern by 
Delaunay triangulation and triangular linear interpolation.

This also tests the faster numba-based option. 

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
import time

from PIL import Image

import context    # Add relative path to get PyBundle

import cv2 as cv
import pybundle

# We load in two images, an image with uniform illumination for calibation
# and an image of a USAF resolution target to demonstrate core removal
calibImg = np.array(Image.open("data\\usaf1_background.tif"))
img = np.array(Image.open("data\\usaf1.tif"))

# Parameters for reconstruction
coreSize = 3          # Estimated core size used when searching for cores
gridSize = 512        # Number of pixels in reconstructed image
filterSize = None     # Pre-Gaussian filter sigma

# One-time calibration
t1 = time.perf_counter()
calib = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, normalise = calibImg,  filterSize = filterSize)
t2 = time.perf_counter()
print('Calibration took:', round(t2-t1,3),'s')

# Image reconstruction without Numba
t1 = time.perf_counter()
imgRecon = pybundle.recon_tri_interp(img, calib, numba = False)
t2 = time.perf_counter()
print('Reconstruction (no numba) took:', round(t2-t1,4),'s')

# Image recostruction with Numba
temp = pybundle.recon_tri_interp(img, calib, numba = True)  # one time init which is slower

t1 = time.perf_counter()
imgRecon = pybundle.recon_tri_interp(img, calib, numba = True)
t2 = time.perf_counter()
print('Reconstruction (numba) took:', round(t2-t1,4),'s')

# Display the cores
plt.figure(dpi = 600)
plt.imshow(calibImg, cmap='gray')
plt.plot(calib.coreX,calib.coreY,'.', markersize = .2, markeredgecolor="r")
plt.title('Core locations')

# Display reconstructed image
plt.figure(dpi = 300)
plt.imshow(imgRecon, cmap='gray')
plt.title('Reconstruction by interpolation')



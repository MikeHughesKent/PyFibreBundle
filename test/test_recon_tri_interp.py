# -*- coding: utf-8 -*-
"""
Test removal of fibre bundle core pattern by 
Delaunay triangulation and triangular linear interpolation.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt
import time

import context    # Add relative path to get PyBundle

import cv2 as cv
import pybundle


# We load in two images, an image with uniform illumination for calibation
# and an image of a USAF resolution target to demonstrate core removal
calibImg = cv.imread("data/usaf1_background.tif")
calibImg = calibImg[:,:,0]

img = cv.imread("data/usaf1.tif")
img = img[:,:,0]


# Parameters for reconstruction
coreSize = 3        # Estimated core size used when searching for cores
gridSize = 400      # Number of pixels in reconstructed image
filterSize = 1      # Pre-Gaussian filter sigma


# One-time calibration
t1 = time.perf_counter()
calib = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, normalise = calibImg,  filterSize = filterSize)
t2 = time.perf_counter()
print('Calibration took:', round(t2-t1,3),' s')


# Image recostruction
t1 = time.perf_counter()
imgRecon = pybundle.recon_tri_interp(img, calib)
t2 = time.perf_counter()
print('Reconstruction took:', round(t2-t1,3),' s')


# Display the cores
plt.figure(dpi = 600)
plt.imshow(calibImg, cmap='gray')
plt.plot(calib.coreX,calib.coreY,'.', markersize = .2, markeredgecolor="r")
plt.title('Reconstruction by interpolation')


# Display reconstructed image
plt.figure(dpi = 300)
plt.imshow(imgRecon, cmap='gray')
plt.title('Core locations')



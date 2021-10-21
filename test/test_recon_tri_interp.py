# -*- coding: utf-8 -*-
"""
Example to demonstrate removal of fibre bundle core pattern by 
Delaunay triangulation and triangular linear interpolation.
Mike Hughes
"""
from matplotlib import pyplot as plt
import time
import context    # Add relative path to get PyBundle


import cv2 as cv
from pybundle import PyBundle


# We load in two images, an image with uniform illumination for calibation
# and an image of a USAF resolution target to demonstrate core removal
calibImg = cv.imread("data/usaf_1_background.tif")
calibImg = calibImg[:,:,0]

img = cv.imread("data/usaf_1.tif")
img = img[:,:,0]


# Find the bundle in the calibration image
loc = PyBundle.findBundle(calibImg)

# Crop background image
calibImgM = PyBundle.mask(calibImg, loc)
calibImgM, newLoc = PyBundle.cropRect(calibImgM,loc)

# Crop test image
imgM = PyBundle.mask(img, loc)
imgM, newLoc = PyBundle.cropRect(imgM,loc)


# Paraneters for reconstruction
coreSize = 6      # Estimated core size used when searching for cores
gridSize = 400    # Number of pixels in reconstructed image
filterSize = 1    # Pre-Gaussian filter sigma
normalise = 1     # Set 1 to normalise w.r.t. core values in calib image


# One-time calibration
t1 = time.perf_counter()
calib = PyBundle.calibTriInterp(calibImgM, coreSize, gridSize, normalise = normalise,  filterSize = filterSize)
t2 = time.perf_counter()
print('Calibration took:', round(t2-t1,3),' s')


# Image recostruction
t1 = time.perf_counter()
imgRecon = PyBundle.reconTriInterp(imgM, calib)
t2 = time.perf_counter()
print('Reconstruction took:', round(t2-t1,3),' s')


# Display the cores
plt.figure(dpi = 600)
plt.imshow(calibImgM, cmap='gray')
plt.plot(calib[0],calib[1],'.', markersize = .2, markeredgecolor="r")


# Display reconstructed image
plt.figure(dpi = 300)
plt.imshow(imgRecon, cmap='gray')


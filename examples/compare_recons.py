# -*- coding: utf-8 -*-
"""
Compares removal of core pattern using Gaussian filter, edge filter and
triangular linear interpolation with various parameters using low level functions.

Images are saved in 'output' folder.

It is recommended for most purposes that the PyBundle class is used instead.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
import os
import time

from PIL import Image
import cv2 as cv

import context    # Add relative path to get pybundle

import pybundle 

# We load in two images, an image with uniform illumination for calibation
# and an image of a USAF resolution target to demonstrate core removal
img = np.array(Image.open("../test/data/usaf1.tif"))
calibImg = np.array(Image.open("../test/data/usaf1_background.tif"))

outFolder = "output"
if not os.path.exists(outFolder):
    os.mkdir(outFolder)

# Parameters for reconstruction
coreSize = 3      # Estimated core size used when searching for cores
gridSize = 400    # Number of pixels in reconstructed image
interpFilterSizes = [None, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]    # Pre-Gaussian filter sigma

# Gaussian filter
gFilterSizes = [0.5,0.75, 1,1.25, 1.5, 1.75, 2,2.25, 2.5,2.75,3]

# Median filter
mFilterSizes = [1,3,5,7,9]

# Edge filter
filtMultipliers = [0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]  # Filters at theses multiple of core spacing
skinThicknesses = [0.1,0.2,0.3,0.4,0.5]            # Controls slope of edge filter


# Find the bundle in the calibration image and generate mask
loc = pybundle.find_bundle(calibImg)
calibImgCrop = pybundle.crop_rect(calibImg, loc)[0]
mask = pybundle.get_mask(calibImg, loc)

# Pre-process the image
img = pybundle.normalise_image(img, calibImg)
img = pybundle.apply_mask(img, mask)
imgCrop = pybundle.crop_rect(img, loc)[0]

# Find core-spacing, used by edge filter
coreSpacing = pybundle.find_core_spacing(calibImgCrop)

# Test Gaussian filter
for gFilterSize in gFilterSizes:
    imgG = pybundle.g_filter(imgCrop, gFilterSize)
    imgSave = Image.fromarray(pybundle.to8bit(imgG))
    imgSave.save(outFolder + '\g_' + str(gFilterSize) + '.tif' )
    
# Test median filter  
for mFilterSize in mFilterSizes:    
    imgM = pybundle.median_filter(pybundle.to8bit(img), mFilterSize)
    imgSave = Image.fromarray(pybundle.to8bit(imgM))
    imgSave.save(outFolder + '\m_' + str(mFilterSize) + '.tif' )

# Test edge filter    
for filtMultiplier in filtMultipliers:
    for skinThickness in skinThicknesses:
        filt = pybundle.edge_filter(np.shape(calibImgCrop)[0], coreSpacing * filtMultiplier, skinThickness)
        imgF = pybundle.filter_image(imgCrop, filt)
        imgSave = Image.fromarray(pybundle.to8bit(imgF))
        imgSave.save(outFolder + '\\f_' + str(filtMultiplier) + "_" + str(skinThickness) + '.tif' )

# Interpolation
for interpFilterSize in interpFilterSizes:
    calib = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, normalise = calibImg,  filterSize = interpFilterSize)
    imgRecon = pybundle.recon_tri_interp(img, calib)
    imgSave = Image.fromarray(pybundle.to8bit(imgRecon))
    imgSave.save(outFolder + '\\interp_' + str(interpFilterSize) + '.tif' )


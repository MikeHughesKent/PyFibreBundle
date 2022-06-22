# -*- coding: utf-8 -*-
"""
Compares removal of core pattern using Gaussina fitler, edge filter and
triangular linear interpolation.
Mike Hughes
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
calibImg = cv.imread("data/usaf1_background.tif")
calibImg = calibImg[:,:,0]

img = cv.imread("data/usaf1.tif")
img = img[:,:,0]

outFolder = "output"

if not os.path.exists(outFolder):
    os.mkdir(outFolder)

# Parameters for reconstruction
coreSize = 3      # Estimated core size used when searching for cores
gridSize = 400    # Number of pixels in reconstructed image
interpFilterSizes = [None, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]    # Pre-Gaussian filter sigma


gFilterSizes = [0.5,0.75, 1,1.25, 1.5, 1.75, 2,2.25, 2.5,2.75,3]
mFilterSizes = [1,3,5,7,9]
skinThicknesses = [0.1,0.2,0.3,0.4,0.5]
filtMultipliers = [0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]

# Find the bundle in the calibration image and crop
loc = pybundle.find_bundle(calibImg)
calibImgCrop = pybundle.crop_rect(calibImg, loc)[0]
coreSpacing = pybundle.find_core_spacing(calibImgCrop)
mask = pybundle.get_mask(calibImg, loc)
img = pybundle.normalise_image(img, calibImg)
img = pybundle.apply_mask(img, mask)
imgCrop = pybundle.crop_rect(img, loc)[0]

# Gaussian Filter
for gFilterSize in gFilterSizes:
    imgG = pybundle.g_filter(imgCrop, gFilterSize)
    imgSave = Image.fromarray(pybundle.to8bit(imgG))
    imgSave.save( outFolder + '\g_' + str(gFilterSize) + '.tif' )
    
# Median Filter  
for mFilterSize in mFilterSizes:    
    imgM = pybundle.median_filter(pybundle.to8bit(img), mFilterSize)
    imgSave = Image.fromarray(pybundle.to8bit(imgM))
    imgSave.save(outFolder + '\m_' + str(mFilterSize) + '.tif' )

# Edge Filter    
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


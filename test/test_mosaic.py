# -*- coding: utf-8 -*-
"""
Basic tests of the Mosaic functionality of PyFibreBundle

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import numpy as np
import math
from matplotlib import pyplot as plt
import time

import cv2 as cv

import context    # For paths to library


import pybundle
from pybundle import Mosaic


filterSize = 1.5    # Size of Gaussian filter to preprocess bundle images


# Load in a fibre bundle endomicroscopy video
cap = cv.VideoCapture('data/raw_example.avi')
ret, img = cap.read()
img = img[:,:,0]
nFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))


# Load in the calibration image
calibImg = cv.imread('data/raw_example_calib.tif')[:,:,0]
loc = pybundle.find_bundle(calibImg)
mask = pybundle.get_mask(calibImg, loc)


# Create mosaic object
mosaic = Mosaic(1000, resize = 250)

# Read in one image and process
ret, img = cap.read()
img = img[:,:,0]
img = pybundle.crop_filter_mask(img, loc, mask, filterSize)
imgStack = np.zeros([nFrames, np.shape(img)[0], np.shape(img)[1] ], dtype='uint8'  ) 
imgStack[0,:,:] = img


# Load video frames
for i in range(1,nFrames):
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, img = cap.read()
    img = img[:,:,0]
    img = pybundle.crop_filter_mask(img, loc, mask, filterSize)
    imgStack[i,:,:] = img

t0 = time.time()


# Do the mosaicing
for i in range(nFrames):
   
    img = imgStack[i,:,:]
    
    mosaic.add(img)   
        
    m = mosaic.get_mosaic()

    cv.imshow('ImageWindow',cv.resize(m,(400,400)).astype('uint8'))
    
    cv.waitKey(1)
        
 
print("Average time to add a frame: " + str(1000 * round( (time.time() - t0)/nFrames,3)) + " ms.")

cv.waitKey(0)
# -*- coding: utf-8 -*-
"""
Full tests of the Mosaic functionality of PyFibreBundle with colour images

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import numpy as np
import math
from matplotlib import pyplot as plt
import time

import cv2 as cv

import context    # For paths to library

import pybundle
from pybundle import PyBundle, Mosaic


filterSize = 1.5    # Size of Gaussian filter to preprocess bundle images

# Load in a fibre bundle endomicroscopy video
cap = cv.VideoCapture('data/bundle_colour_1_video.avi')
ret, img = cap.read()
img = img[:,:,::-1]
nFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))


# Load in the calibration image
calibImg = cv.imread('data/bundle_colour_1_background.tif')
calibImg = calibImg[:,:,::-1]

# Create PyBundle object for core removal
pyb = PyBundle(coreMethod = PyBundle.TRILIN,  # Set to remove core pattern by trianglar linear interpolation
               coreSize = 3,                  # Providing an estimate of the core spacing in pixels help to identify core locations robustly
               calibImage = calibImg,
               normaliseImage = calibImg,
               autoContrast = False)
pyb.calibrate()


# Create mosaic object
mosaic = Mosaic(1000, resize = 250)

# Read in one image and process
ret, img = cap.read()
img = img[:,:,::-1]
imgProc = pyb.process(img)
imgStack = np.zeros([nFrames, 512, 512, 3 ], dtype='uint16' ) 
imgStack[0,:,:,:] = imgProc


# Load video frames
for i in range(1,nFrames):
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, img = cap.read()
    img = img[:,:,::-1]

    img = pyb.process(img)
    imgStack[i,:,:,:] = img 


# Test routine. Once a mosaic object is created with the desired parameters
# it is passed here as 'mosaic' with a 'description' used to label the output
def test_mosaic(mosaic, description):
   
    mosaic.reset()
    t0 = time.time()
    for i in range(nFrames):   
        img = imgStack[i,:,:,:]
       
        mosaic.add(img)   
        mosaicImage = mosaic.get_mosaic()

    plt.figure()
    plt.imshow(mosaicImage / np.max(mosaicImage), cmap = 'gray') 
    plt.title(description)
 
    print(f"{description}: Average time to add a frame to mosaic: {str(1000 * round( (time.time() - t0)/nFrames,3))} ms.")


# The tests

mosaic = Mosaic(1000, blend = True)
test_mosaic(mosaic, "Default")

mosaic = Mosaic(1000, blend = False)
test_mosaic(mosaic, "Default, no blend")

mosaic = Mosaic(1000, resize = 250)
test_mosaic(mosaic, "Default with resize to 250")

mosaic = Mosaic(1000, resize = 250, blend = False)
test_mosaic(mosaic, "No Blend")

mosaic = Mosaic(1000, resize = 250, blend = True)
test_mosaic(mosaic, "Blend")

mosaic = Mosaic(1000, resize = 250, blend = True, blendDist = 5)
test_mosaic(mosaic, "Blend distance of 5 px")

mosaic = Mosaic(500, resize = 250)
test_mosaic(mosaic, "Crop at Edge (Default)")

mosaic = Mosaic(500, resize = 250, boundaryMethod = mosaic.EXPAND)
test_mosaic(mosaic, "Expand at Edge")

mosaic = Mosaic(500, resize = 250, boundaryMethod = mosaic.SCROLL)
test_mosaic(mosaic, "Scroll at Edge")

mosaic = Mosaic(1000, resize = 250, resetThresh = .985)
test_mosaic(mosaic, "Reset on threshold")

mosaic = Mosaic(1000, resize = 250, resetIntensity = 80)
test_mosaic(mosaic, "Reset on intensity")

mosaic = Mosaic(1000, resize = 250, resetSharpness = 2)
test_mosaic(mosaic, "Reset on sharpness")

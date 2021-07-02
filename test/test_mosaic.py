# -*- coding: utf-8 -*-
"""
Some basic tests of the Mosaic functionality of PyBundle

Mike Hughes, Applied Optics Group, University of Kent

"""


import context
import numpy as np
import math
from matplotlib import pyplot as plt
import time

import cv2 as cv

from pybundle import PyBundle
from pybundle import Mosaic

# Load in a fibre bundle endomicroscopy video
cap = cv.VideoCapture('data/raw_example.avi')
ret, img = cap.read()
img = img[:,:,0]
nFrames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))


# Load in the calibration image
calibImg = cv.imread('data/raw_example_calib.tif')[:,:,0]
loc, mask = PyBundle.locateBundle(calibImg)


# Create mosaic object
mosaic = Mosaic(1000, resize = 250)

# Read in one image and process
ret, img = cap.read()
img = img[:,:,0]
img = PyBundle.cropFilterMask(img, loc, mask, 1.5)
imgStack = np.zeros([nFrames, np.shape(img)[0],np.shape(img)[1] ], dtype='uint8'  ) 
imgStack[0,:,:] = img

# Load video frames
for i in range(1,nFrames):
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, img = cap.read()
    img = img[:,:,0]
    img = PyBundle.cropFilterMask(img, loc, mask, 1.5)
    imgStack[i,:,:] = img

t0 = time.time()

# Do the mosaicing
for i in range(nFrames):
   
    img = imgStack[i,:,:]
    
    mosaic.add(img)   
        
    m = mosaic.getMosaic()

    cv.imshow('ImageWindow',cv.resize(m,(400,400)).astype('uint8'))
    cv.waitKey(1);
        
 
print((time.time() - t0)/nFrames)


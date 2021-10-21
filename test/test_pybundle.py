# -*- coding: utf-8 -*-
"""
Some general tests of PyBundle

@author: Mike Hughes
"""
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import math
import cv2 as cv
import time

import context
from pybundle import PyBundle

img = cv.imread("data/bundle1.tif")
img = img[:,:,0]

# Find bundle, crop image to bundle and generate mask
loc = PyBundle.findBundle(img)
imgD, croppedLoc = PyBundle.cropRect(img,loc)
mask = PyBundle.getMask(imgD, croppedLoc)


nTests = 100
filterSize = 2.5

t1 = time.time()


for i in range(nTests):

   
    imgProc = PyBundle.cropFilterMask(img, loc, mask, filterSize)
    
    # Equivalently can call
    #imgProc, newLoc = PyBundle.cropRect(img,loc)
    #imgProc = PyBundle.gFilter(imgProc, 2.5)
    #imgProc= PyBundle.mask(imgProc, mask)

t2 = time.time()

print("Processing Time (ms): ", round(1000 * (t2-t1) / nTests,2))

plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')

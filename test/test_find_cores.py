# -*- coding: utf-8 -*-
"""
Tests the find_cores function of PyBundle

@author: Mike Hughes
"""

from matplotlib import pyplot as plt
import sys
import cv2 as cv
import context    # Add relative path to get PyBundle
from pybundle import PyBundle



img = cv.imread("data/bundle1.tif")
img = img[:,:,0]
loc = PyBundle.findBundle(img)
imgM = PyBundle.mask(img, loc)
imgM, newLoc = PyBundle.cropRect(imgM,loc)
iz= PyBundle.gFilter(imgM, 3)
cx,cy = PyBundle.findCores(imgM, 6)
plt.figure(dpi=600)
plt.imshow(imgM, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)


img = cv.imread("data/bundle2.tif")
img = img[:,:,0]
loc = PyBundle.findBundle(img)
imgM = PyBundle.mask(img, loc)
imgM, newLoc = PyBundle.cropRect(imgM,loc)
cx,cy = PyBundle.findCores(imgM, 8)
plt.figure(dpi=600)
plt.imshow(imgM, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)


img = cv.imread("data/bundle3.tif")
img = img[:,:,0]
loc = PyBundle.findBundle(img)
imgM = PyBundle.mask(img, loc)
imgM, newLoc= PyBundle.cropRect(imgM,loc)
cx,cy = PyBundle.findCores(imgM, 6)
plt.figure(dpi=600)
plt.imshow(imgM, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)



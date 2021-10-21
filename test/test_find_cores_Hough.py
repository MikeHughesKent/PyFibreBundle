# -*- coding: utf-8 -*-
"""
Tests the find_cores_Hough function of PyBundle

@author: Mike Hughes
"""
#from pybundle import PyBundle

from matplotlib import pyplot as plt

import cv2 as cv

import context
from pybundle import PyBundle

img = cv.imread("data/bundle1.tif")
img = img[:,:,0]
loc = PyBundle.findBundle(img)
imgM = PyBundle.mask(img, loc)
imgM, newLoc = PyBundle.cropRect(imgM,loc)
cx,cy, imgF, edges, circs = PyBundle.findCoresHough(imgM, darkRemove = 2, estRad = 1, scaleFactor = 3)
plt.figure(dpi=600)
plt.imshow(imgM, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)


img = cv.imread("data/bundle3.tif")
img = img[:,:,0]
loc = PyBundle.findBundle(img)
imgM = PyBundle.mask(img, loc)
imgM, newLoc = PyBundle.cropRect(imgM,loc)
cx,cy, imgF, edges, circs = PyBundle.findCoresHough(imgM, estRad = 2, scaleFactor = 2)
plt.figure(dpi=600)
plt.imshow(imgM, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)


img = cv.imread("data/bundle3.tif")
img = img[:,:,0]
loc = PyBundle.findBundle(img)
imgM = PyBundle.mask(img, loc)
imgM, newLoc = PyBundle.cropRect(imgM,loc)
cx,cy, imgF, edges, circs = PyBundle.findCoresHough(imgM, darkRemove = 3, estRad = 1.5, scaleFactor = 3)
plt.figure(dpi=600)
plt.imshow(img, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)






#bundle = PyBundle()

#bundle.gaussianFilter()

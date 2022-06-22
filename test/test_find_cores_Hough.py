# -*- coding: utf-8 -*-
"""
Tests the find_cores_Hough function of pybundle.

Note than find_cores is faster and usually more accurate.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

from matplotlib import pyplot as plt

import cv2 as cv

import context    # For paths to library

import pybundle

img = cv.imread("data/bundle1.tif")
img = img[:,:,0]
imgMasked = pybundle.auto_mask(img)
cx,cy, = pybundle.find_cores_hough(imgMasked, darkRemove = 2, estRad = 1, scaleFactor = 3)
plt.figure(dpi=600)
plt.imshow(imgMasked, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)
print("Found " + str(len(cx)) + " cores.")


img = cv.imread("data/bundle3.tif")
img = img[:,:,0]
imgMasked = pybundle.auto_mask(img)
cx,cy = pybundle.find_cores_hough(imgMasked, estRad = 2, scaleFactor = 2)
plt.figure(dpi=600)
plt.imshow(imgMasked, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)
print("Found " + str(len(cx)) + " cores.")


img = cv.imread("data/bundle3.tif")
img = img[:,:,0]
imgMasked = pybundle.auto_mask(img)
cx,cy = pybundle.find_cores_hough(imgMasked, darkRemove = 3, estRad = 1.5, scaleFactor = 3)
plt.figure(dpi=600)
plt.imshow(imgMasked, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)
print("Found " + str(len(cx)) + " cores.")



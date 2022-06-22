# -*- coding: utf-8 -*-
"""
Some general tests of PyBundle using object-oriented programming


@author: Mike Hughes
Applied Optics Group
University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import math
import cv2 as cv
import time

import context
import pybundle
from pybundle import PyBundle

img = cv.imread("data/usaf1.tif")
img = img[:,:,0]
calibImg = cv.imread("data/usaf1_background.tif")
calibImg = calibImg[:,:,0]



# Create an instance of PyBundle object
pyb = PyBundle()



# Recon using Gaussian filter
pyb.set_auto_loc(img)
pyb.set_auto_mask(img)
pyb.set_core_method(pyb.FILTER)
pyb.set_output_type('uint8')
pyb.set_auto_contrast(True)
pyb.set_filter_size(2.5)
pyb.set_crop(True)

imgProc = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Gaussian Filter')




# Recon using a Gaussian filter, specify a specific bundle radius
pyb.set_auto_mask(calibImg, radius = 300)
imgProcRadius = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProcRadius, cmap='gray')
plt.title('Gaussian Filter, small radius')

pyb.set_auto_mask(calibImg)   # Set mask back to full image





# Recon using Gaussian filter and obtain a 16 bit output
pyb.set_output_type('uint16')
imgProc16 = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProc16, cmap='gray')
plt.title('Gaussian Filter, 16 bit')

pyb.set_output_type('uint8')   # Set output back to 8 bit



# Recon using edge filter
pyb.set_core_method(pyb.EDGE_FILTER)
pyb.set_auto_loc(calibImg)
pyb.set_crop(True)
coreSpacing = pybundle.find_core_spacing(calibImg)
pyb.set_edge_filter(coreSpacing * 1.8, coreSpacing * 0.2)
imgProc = pyb.process(img)


plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Edge filter')





# Triangular Linear Interpolation with normalisation
pyb.set_core_method(pyb.TRILIN)
pyb.set_calib_image(calibImg)
pyb.set_grid_size(512)
pyb.set_normalise_image(calibImg)
pyb.calibrate()
pyb.set_auto_contrast(True)

imgProc = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Tri Lin Interp, normalisation')




# Triangular Linear Interpolation with no normalisation
pyb.set_normalise_image(None)
imgProc = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Tri Lin Interp, no normalisation')

pyb.set_normalise_image(calibImg)   # Put the normalisation back in




# Triangular Linear Interpolation with auto contrast off, but output turned
# to float. This is needed if 'normalise image' is used as otherwise the 
pyb.set_auto_contrast(False)
pyb.set_output_type('float')

imgProc = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Tri Lin Interp, not a.c., float')
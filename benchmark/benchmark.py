# -*- coding: utf-8 -*-
"""

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
import os
import time

from PIL import Image
import cv2 as cv

import context    # Add relative path to get pybundle

from pybundle import PyBundle 
import pybundle

from pathlib import Path

import timeit


from IPython import get_ipython


# We load in two images, an image with uniform illumination for calibation
# and an image of a USAF resolution target to demonstrate core removal
img = np.array(Image.open(Path('../test/data/tissue_paper.tif')))
calibImg = np.array(Image.open(Path('../test/data/tissue_paper_back.tif')))

outFolder = "output"
if not os.path.exists(outFolder):
    os.mkdir(outFolder)

# Parameters for reconstruction

pyb = PyBundle(coreMethod = PyBundle.TRILIN, gridSize = 512, 
               calibImage = calibImg, backgroundImage = calibImg, 
               normaliseImage = calibImg)

print("Linear Interp, 512 grid, norm, calibration")   
get_ipython().magic('%timeit pyb.calibrate()')

print("Linear Interp, 512 grid, norm, recon")   
get_ipython().magic('%timeit pyb.process(img)')





pyb = PyBundle(coreMethod = PyBundle.TRILIN, gridSize = 1024, 
               calibImage = calibImg, backgroundImage = calibImg,
               normaliseImage = calibImg)

print("Linear Interp, 1024 grid, norm, calibration")   
get_ipython().magic('%timeit pyb.calibrate()')

print("Linear Interp, 1024 grid, norm, recon")   
get_ipython().magic('%timeit pyb.process(img)')



pyb = PyBundle(coreMethod = PyBundle.FILTER, calibImage = calibImg, 
               filterSize = 2)
pyb.calibrate()

print("Gaussian, 2 px sigma")   
get_ipython().magic('%timeit pyb.process(img)')





pyb = PyBundle(coreMethod = PyBundle.FILTER, calibImage = calibImg, 
               filterSize = 2)
pyb.calibrate()

print("Gaussian, 2 px sigma")   
get_ipython().magic('%timeit pyb.process(img)')



pyb = PyBundle(coreMethod = PyBundle.FILTER, calibImage = calibImg, 
               filterSize = 2, background = calibImg)
pyb.calibrate()

print("Gaussian, 2 px sigma, back")   
get_ipython().magic('%timeit pyb.process(img)')




pyb = PyBundle(coreMethod = PyBundle.FILTER, calibImage = calibImg, 
               filterSize = 2, normaliseImage = calibImg, background = calibImg)
pyb.calibrate()

print("Gaussian, 2 px sigma, norm and back")   
get_ipython().magic('%timeit pyb.process(img)')



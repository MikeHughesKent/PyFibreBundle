# -*- coding: utf-8 -*-
"""
Times several key functions of PyFibreBundle.

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


""" Linear Interpolation """
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


""" Gaussian Filter """

pyb = PyBundle(coreMethod = PyBundle.FILTER, calibImage = calibImg, crop = True,
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


""" Super Resolution"""

dataFolder = Path('../test/data/super_res/data')
backFile =  Path('../test/data/super_res/background.tif')

nImages = 8        # Means all files in folder will be used
coreSize = 3       # Estimate, used by core finding
gridSize = 800     # Reconstruction grid size
shift = None       # If shifts are known, can specify them here
filterSize = None  # Filter applied prior to extracting core values

# Find images in folder
files = [f.path for f in os.scandir(dataFolder)]

if nImages is None:
    nImages = len(files)

# Load images
img = np.array(Image.open(files[0]))
imSize = np.shape(np.array(img))
imgs = np.zeros((imSize[0], imSize[1],nImages), dtype='uint8')

for idx, fName in enumerate(files[:nImages]):
    img = Image.open(fName)
    imgs[:,:,idx] = np.array(img)

calibImg = np.array(Image.open(backFile))

pyb = PyBundle(coreMethod = PyBundle.TRILIN, superRes = True, gridSize = gridSize, autoContrast = False, 
               calibImage = calibImg, normaliseImage = calibImg, 
               backgroundImage = calibImg, srCalibImages = imgs)

print("Super Res Calibration")   
t1 = time.perf_counter()
pyb.calibrate_sr()
print(round((time.perf_counter() - t1) * 1000))
print("Super Res Recon")   
get_ipython().magic('%timeit pyb.process(imgs)')



""" Mosaicing """



# -*- coding: utf-8 -*-
"""
Example of use super-resolution functionality of PyFibreBundle using PyBundle Class.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import context    # For paths to library

import pybundle
from pybundle import PyBundle

dataFolder = r"..\test\data\super_res\data"
backFile =  r"..\test\data\super_res\background.tif"

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

pyb = PyBundle()

pyb.set_core_method(pyb.TRILIN)
pyb.set_grid_size(gridSize)
pyb.set_auto_contrast(True)
pyb.set_calib_image(calibImg)
pyb.set_background(calibImg)
pyb.set_normalise_image(calibImg)


""" Single image recon for comparison """
pyb.calibrate()
reconSingle = pyb.process(imgs[:,:,0])

""" Super Resolution Recon """
pyb.set_super_res(True)
pyb.set_sr_calib_images(imgs)
pyb.calibrate_sr()
reconSR = pyb.process(imgs)

plt.figure(dpi = 150)
plt.imshow(reconSingle, cmap='gray')
plt.title('Single Image')

plt.figure(dpi = 150)
plt.imshow(reconSR, cmap='gray')
plt.title('SR Image')


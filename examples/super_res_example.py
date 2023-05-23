# -*- coding: utf-8 -*-
"""
Example of use super-resolution functionality of PyFibreBundle using PyBundle Class.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import context    # For paths to library

import pybundle
from pybundle import PyBundle

dataFolder = Path('../test/data/super_res/data')
backFile =  Path('../test/data/super_res/background.tif')

shift = None       # If shifts are known, can specify them here
nImages = None

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


""" Single image recon for comparison """

pyb = PyBundle(coreMethod = PyBundle.TRILIN,  # Set to remove core pattern by trianglar linear interpolation
               gridSize = 800,                # Size of output image
               coreSize = 3,                  # Providing an estimate of the core spacing in pixels help to identify core locations robustly
               calibImage = calibImg, 
               normaliseImage = calibImg)

pyb.calibrate()
reconSingle = pyb.process(imgs[:,:,0])

plt.figure(dpi = 150)
plt.imshow(reconSingle, cmap='gray')
plt.title('Single Image')




""" Super Resolution Recon """

pyb = PyBundle(coreMethod = PyBundle.TRILIN,  # Set to remove core pattern by trianglar linear interpolation
               gridSize = 800,                # Size of output image
               coreSize = 3,                  # Providing an estimate of the core spacing in pixels help to identify core locations robustly
               calibImage = calibImg, 
               normaliseImage = calibImg,
               superRes = True,               # Set to True to do Super Res
               srCalibImages = imgs)          # These are the shifted images


pyb.calibrate_sr()
reconSR = pyb.process(imgs)

plt.figure(dpi = 150)
plt.imshow(reconSR, cmap='gray')
plt.title('Resolution Enhanced Image')

plt.show()
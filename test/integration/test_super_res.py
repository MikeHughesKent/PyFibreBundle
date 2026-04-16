# -*- coding: utf-8 -*-
"""
Tests super-resolution reconstruction of PyFibreBundle.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

import context    # For paths to library

import pybundle
from pybundle import SuperRes


dataFolder = r"data\super_res\data"
backFile =  r"data\super_res\background.tif"

nImages = 8        # Means all files in folder will be used
coreSize = 3       # Esimate, used by core finding
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

 
""" Single image recon for comparison """
calibSingle = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, filterSize = filterSize, background = calibImg, normalise = calibImg, autoMask = True)
reconSingle = pybundle.recon_tri_interp(imgs[:,:,0], calibSingle)

plt.figure(dpi = 150)
plt.imshow(reconSingle, cmap='gray')
plt.title('Single Image')
 

""" Super Resolution Recon """
t1 = time.perf_counter()
calib = SuperRes.calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, filterSize = filterSize, normalise = calibImg, autoMask = True)
print("Calibration time:", round(time.perf_counter() - t1, 3))

t1 = time.perf_counter()
reconImg = SuperRes.recon_multi_tri_interp(imgs, calib)
print("Reconstruction time:", round(time.perf_counter() - t1, 3))

plt.figure(dpi = 150)
plt.imshow(reconImg, cmap='gray')
plt.title('SR Image')


""" Super Resolution Recon with Intensity Normalisation"""
t1 = time.perf_counter()
calib = SuperRes.calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, filterSize = filterSize, normalise = calibImg, normToImage = True)
print("Calibration time:", round(time.perf_counter() - t1, 3))

t1 = time.perf_counter()
reconImg = SuperRes.recon_multi_tri_interp(imgs, calib)
print("Reconstruction time:", round(time.perf_counter() - t1, 3))

plt.figure(dpi = 150)
plt.imshow(reconImg, cmap='gray')
plt.title('SR Image with Normalisation')


""" Providing shifts instead of using images """
shifts = np.array([[  0.        ,   0.        ],
                   [-20.15261569,  -2.30315608],
                   [-24.75892784, -12.66735843],
                   [-28.21366196, -21.87998274],
                   [ -2.8789451 , -10.93999137],
                   [ -4.60631216, -16.69788157],
                   [ -9.78841333, -30.51681804],
                   [-17.27367059, -27.06208392]])
t1 = time.perf_counter()
calib = SuperRes.calib_multi_tri_interp(calibImg, None, coreSize, gridSize,  shifts = shifts, filterSize = filterSize, normalise = calibImg)
print(f"Calibration time: {round(time.perf_counter() - t1, 3)} s")

t1 = time.perf_counter()
reconImg = SuperRes.recon_multi_tri_interp(imgs, calib)
print(f"Reconstruction time: {round(time.perf_counter() - t1, 3)} s")

plt.figure(dpi = 150)
plt.imshow(reconImg, cmap='gray')
plt.title('SR with specified shifts')
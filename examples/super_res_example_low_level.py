# -*- coding: utf-8 -*-
"""
Example of using super-resolution functionaity of PyFibreBundle to enhance resolution
by combining shifted images.

For most purposes the pybundle class should be used instead
- see super_res_example.py

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

import context    # For paths to library

import pybundle
from pybundle import SuperRes

dataFolder = Path('../test/data/super_res/data')
backFile =  Path('../test/data/super_res/background.tif')

nImages = 4        # We will use just four of the images from the folder 
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

 
""" Single image recon for comparison """
calibSingle = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, 
                                        filterSize = filterSize, normalise = calibImg, 
                                        mask = True, autoMask = True)

reconSingle = pybundle.recon_tri_interp(imgs[:,:,0], calibSingle)

plt.figure(dpi = 150)
plt.imshow(reconSingle, cmap='gray')
plt.title('Single Image')
 

""" Super Resolution Recon with Intensity Normalisation"""
t1 = time.perf_counter()
calib = SuperRes.calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, 
                                        filterSize = filterSize, normalise = calibImg,
                                        mask = True, autoMask = True)

print(f"Calibration time: {round(time.perf_counter() - t1, 3)}")

t1 = time.perf_counter()
reconImg = SuperRes.recon_multi_tri_interp(imgs, calib)
print(f"Reconstruction time:{round(time.perf_counter() - t1, 3)}")

plt.figure(dpi = 150)
plt.imshow(reconImg, cmap='gray')
plt.title('Resolution Enhanced Image')

plt.show()
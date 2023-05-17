# -*- coding: utf-8 -*-
"""
Simple example of how to use PyFibreBundle to remove core pattern with filtering.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from time import perf_counter as timer

from pathlib import Path

import context

from pybundle import PyBundle

# Load images
img = np.array(Image.open(Path('../test/data/usaf1.tif')))
calibImg = np.array(Image.open(Path('../test/data/usaf1_background.tif')))

# Create an instance of the PyBundle class, set to remove core pattern by Gaussian filtering only
pyb = PyBundle(coreMethod = PyBundle.FILTER, 
               filterSize = 2.5)

t1 = timer()
imgProc = pyb.process(img)
print(f"Gaussian filter took {round((timer() - t1) * 1000)} ms ")

# Create an instance of the PyBundle class, set to remove core pattern by Gaussian filtering, crop and mask
pyb = PyBundle(coreMethod = PyBundle.FILTER, 
               filterSize = 2.5,
               applyMask = True)

t1 = timer()
imgProc2 = pyb.process(img)
print(f"Gaussian filter and crop/mask from image took {round((timer() - t1) * 1000)} ms ")



# Create an instance of the PyBundle class, set to remove core pattern by Gaussian filtering, crop and mask
# based on a calib image
pyb = PyBundle(coreMethod = PyBundle.FILTER, 
               filterSize = 2.5,
               crop = True,
               applyMask = True,
               calibImage = calibImg)
pyb.calibrate()

t1 = timer()
imgProc3 = pyb.process(img)
print(f"Gaussian filter and crop/mask from pre-calib took {round((timer() - t1) * 1000)} ms ")



plt.figure(dpi=150)
plt.imshow(img, cmap='gray')
plt.title("Raw image")

plt.figure(dpi=150)
plt.imshow(imgProc, cmap='gray')
plt.title("Gaussian filter")

plt.figure(dpi=150)
plt.imshow(imgProc2, cmap='gray')
plt.title("Gaussian filter, mask and crop from image")

plt.figure(dpi=150)
plt.imshow(imgProc3, cmap='gray')
plt.title("Gaussian filter, mask and crop from calib image")

plt.show()

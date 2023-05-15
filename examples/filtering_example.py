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

from pathlib import Path

import context

from pybundle import PyBundle

# Load images
img = np.array(Image.open(Path('../test/data/usaf1.tif')))
calibImg = np.array(Image.open(Path('../test/data/usaf1_background.tif')))

# Create an instance of the PyBundle class, set to remove core pattern by Gaussian filtering and 
# crop square around bundle
pyb = PyBundle(coreMethod = PyBundle.FILTER, 
               filterSize = 2.5, 
               crop = True)

# Automatically create a mask around bundle using the calibration image
pyb.create_and_set_mask(calibImg)

# Do core removal, masking and cropping
imgProc = pyb.process(img)


plt.figure(dpi=150)
plt.imshow(img, cmap='gray')
plt.title("Raw image")

plt.figure(dpi=150)
plt.imshow(imgProc, cmap='gray')
plt.title("Cropped image with filter applied")

plt.show()

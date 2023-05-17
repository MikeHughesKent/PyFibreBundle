# -*- coding: utf-8 -*-
"""
Simple example of how to use PyFibreBundle to remove core patterns from
colour image with Gaussian and edge filters.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from time import perf_counter as timer

from pathlib import Path

import context

from pybundle import PyBundle

# Load images
img = np.array(Image.open(r"..\test\data\bundle_colour_1.tif"))
calibImg = np.array(Image.open(r"..\test\data\bundle_colour_1_background.tif"))




# Create an instance of the PyBundle class, set to remove core pattern by Gaussian filtering, crop and mask
# based on a calib image
pyb = PyBundle(coreMethod = PyBundle.FILTER, 
               filterSize = 1.6,
               crop = True,
               applyMask = True,
               calibImage = calibImg)
pyb.calibrate()

t1 = timer()
imgProc = pyb.process(img)
print(f"Colour Gaussian filter took {round((timer() - t1) * 1000)} ms ")

plt.figure(dpi=300)
plt.imshow(imgProc / 1024, cmap='gray')
plt.title("Gaussian filter")




# Create an instance of the PyBundle class, set to remove core pattern by Edge filtering using
# calibration image to pre-determine the size of bundle and pre-calculating the filter
pyb = PyBundle(coreMethod = PyBundle.EDGE_FILTER, 
               edgeFilterShape = (5.6,1), 
               crop = True,
               applyMask = True,
               calibImage = calibImg) 
               
pyb.calibrate()
t1 = timer()
imgProc2 = pyb.process(img)
print(f"Colour edge filter took {round((timer() - t1) * 1000)} ms ")

plt.figure(dpi=300)
plt.imshow(imgProc2 / 1024, cmap='gray')
plt.title("Colour Edge filter")



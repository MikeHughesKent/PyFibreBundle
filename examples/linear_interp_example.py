# -*- coding: utf-8 -*-
"""
Simple example of how to use PyFibreBundle to remove core pattern.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import time

import context

from pybundle import PyBundle

img = np.array(Image.open("../test/data/usaf1.tif"))
calibImg = np.array(Image.open("../test/data/usaf1_background.tif"))

# Create an instance of the PyBundle class 
pyb = PyBundle()

# Set to remove core pattern by trianglar linear interpolation
pyb.set_core_method(pyb.TRILIN)

# Provide the calibration and normalisation images
pyb.set_calib_image(calibImg)
pyb.set_normalise_image(calibImg)

# We call this now to do the calibration. This is the time consuming step. Otherwise it will be done when we called process.
t1 = time.perf_counter()
pyb.calibrate()
print(f"One-time calibration took {round(1000 * (time.perf_counter() - t1), 1) } ms.")

# Do core removal
t1 = time.perf_counter()
imgProc = pyb.process(img)
print(f"Reconstruction took {round(1000 * (time.perf_counter() - t1), 1) } ms.")

plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
# -*- coding: utf-8 -*-
"""
Simple example of how to use PyBundle class f PyFibreBundle to remove core pattern.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import time

from pathlib import Path

import context

from pybundle import PyBundle

# Load images
img = np.array(Image.open(Path('../test/data/usaf1.tif')))
calibImg = np.array(Image.open(Path('../test/data/usaf1_background.tif')))

# Create an instance of the PyBundle class 
pyb = PyBundle(coreMethod = PyBundle.TRILIN,  # Set to remove core pattern by trianglar linear interpolation
               coreSize = 3,                  # Providing an estimate of the core spacing in pixels help to identify core locations robustly
               calibImage = calibImg,
               normaliseImage = calibImg)

# We call this now to do the calibration. This is the time-consuming step. Otherwise it will be done when we called process.
t1 = time.perf_counter()
pyb.calibrate()
print(f"One-time calibration took {round(1000 * (time.perf_counter() - t1), 1) } ms.")

# Do core removal
imgProc = pyb.process(img)         # Do this once to initialise Numba JIT so later timing is accurate

t1 = time.perf_counter()
imgProc = pyb.process(img)
print(f"Reconstruction took {round(1000 * (time.perf_counter() - t1), 1) } ms.")

plt.figure(dpi=150)
plt.imshow(imgProc, cmap='gray')
plt.title('Reconstructed Image')

plt.show()
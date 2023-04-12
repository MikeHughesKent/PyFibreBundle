# -*- coding: utf-8 -*-
"""
Example of how to use Numba to accelerate PyFibreBundle removing core pattern by 
linear interpolation. The first reconstruction is slower because of the need to run the JIT,
subsequent calls are faster by typically 3X.

@author: Mike Hughes, Applied Optics Group, University of Kent
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
pyb = PyBundle(coreMethod = PyBundle.TRILIN,  # Set to remove core pattern by trianglar linear interpolation
               coreSize = 3,                  # Providing an estimate of the core spacing in pixels help to identify core locations robustly
               gridSize = 512,
               calibImage = calibImg,
               normaliseImage = calibImg)


# Disable Numba
pyb.set_use_numba(False)

# We call this now to do the calibration. This is the time consuming step. Otherwise it will be done when we called process.
t1 = time.perf_counter()
pyb.calibrate()
print(f"One-time calibration took {round(1000 * (time.perf_counter() - t1), 1) } ms.")

# Do core removal without Numba
t1 = time.perf_counter()
imgProc = pyb.process(img)
print(f"Reconstruction without Numba took {round(1000 * (time.perf_counter() - t1), 1) } ms.")

# Now we use Numba
pyb.set_use_numba(True)

t1 = time.perf_counter()
imgProcNumba = pyb.process(img)
print(f"Reconstruction of the first image with Numba took {round(1000 * (time.perf_counter() - t1), 1) } ms.")

t1 = time.perf_counter()
imgProcNumba = pyb.process(img)
print(f"Reconstruction of subsequent images with Numba took {round(1000 * (time.perf_counter() - t1), 1) } ms.")

plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title("No Numba")

plt.figure(dpi=300)
plt.imshow(imgProcNumba, cmap='gray')
plt.title("Numba")

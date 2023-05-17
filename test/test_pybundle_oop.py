# -*- coding: utf-8 -*-
"""
Some general tests of PyBundle using object-oriented programming

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
import time

from PIL import Image

import context
import pybundle

from pybundle import PyBundle

img = np.array(Image.open("data/usaf1.tif"))
calibImg = np.array(Image.open("data/usaf1_background.tif"))

pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2.5, outputType = 'uint8', 
               autoContrast = True, crop = True, autoMask = True)

pyb.set_auto_loc(img)


t1 = time.perf_counter()
imgProc = pyb.process(img)
print(f"Time for Gaussian filter processing: {round(time.perf_counter() - t1,4)} s")

plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Gaussian Filter')


# Recon using a Gaussian filter, specify a specific bundle radius
pyb.set_auto_mask(calibImg, radius = 300)
imgProcRadius = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProcRadius, cmap='gray')
plt.title('Gaussian Filter, small radius')


pyb.set_auto_mask(calibImg)   # Set mask back to full image


# Recon using Gaussian filter and obtain a 16 bit output
pyb.set_output_type('uint16')
imgProc16 = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProc16, cmap='gray')
plt.title('Gaussian Filter, 16 bit')

pyb.set_output_type('uint8')   # Set output back to 8 bit



# Recon using edge filter
pyb.set_core_method(pyb.EDGE_FILTER)
pyb.set_auto_loc(calibImg)
pyb.set_crop(True)
coreSpacing = pybundle.find_core_spacing(calibImg)
pyb.set_edge_filter_shape(coreSpacing * 1.8, coreSpacing * 0.2)
t1 = time.perf_counter()
imgProc = pyb.process(img)
print(f"Time for edge filter processing: {round(time.perf_counter() - t1,4)} s")

plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Edge filter')



# Triangular Linear Interpolation with normalisation
pyb.set_core_method(pyb.TRILIN)
pyb.set_calib_image(calibImg)
pyb.set_grid_size(512)
pyb.set_normalise_image(calibImg)

t1 = time.perf_counter()
pyb.calibrate()
print(f"Time for tri linear interp calibration: {round(time.perf_counter() - t1,4)} s")

pyb.set_auto_contrast(True)
pyb.set_use_numba(False)

t1 = time.perf_counter()
imgProc = pyb.process(img)
print(f"Time for tri linear interp processing: {round(time.perf_counter() - t1,4)} s")
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Tri Lin Interp, normalisation')




# Triangular Linear Interpolation with no normalisation
pyb.set_normalise_image(None)
imgProc = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Tri Lin Interp, no normalisation')


pyb.set_normalise_image(calibImg)   # Put the normalisation back in

# Triangular Linear Interpolation with auto contrast off, but output turned
# to float. This is needed if 'normalise image' is used.
pyb.set_auto_contrast(False)
pyb.set_output_type('float')
pyb.set_use_numba(True)

imgProc = pyb.process(img)
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Tri Lin Interp, not a.c., float')


t1 = time.perf_counter()
imgProc = pyb.process(img)
print(f"Time for tri linear interp processing with numba: {round(time.perf_counter() - t1,4)} s")
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Tri Lin Interp, not a.c., float, numba')
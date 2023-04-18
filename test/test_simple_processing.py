# -*- coding: utf-8 -*-
"""
Some general tests of simple processing in pybundle

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
import time
from PIL import Image
import context

import pybundle 

filterSize = 2.5

img = np.array(Image.open("data/usaf1.tif"))
calibImg = np.array(Image.open("data/usaf1_background.tif"))

# Locate the bundle
loc = pybundle.find_bundle(calibImg)

# Generate a mask by specifying the location of the bundle
mask = pybundle.get_mask(calibImg, loc)


# Estimate the core spacing
coreSpacing = pybundle.find_core_spacing(calibImg)


# Produce an image by Gaussian filtering, masking and cropping
t1 = time.time()
imgProc = pybundle.g_filter(img, filterSize)
imgProc = pybundle.apply_mask(imgProc, mask)
imgProc = pybundle.crop_rect(imgProc, loc)[0]
t2 = time.time()

print(f"Gaussian Filter Processing Time (ms): { round(1000 * (t2-t1),2)}" )
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Sequential G Filter, mask, crop')
plt.show()


# Produce an image using the simple Gaussian filtering, masking and cropping
t1 = time.time()
imgProc = pybundle.crop_filter_mask(img, loc, mask, filterSize)
t2 = time.time()

print(f"Simple Gaussian Processing Time (ms): { round(1000 * (t2-t1),2)}" )
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Combined G Filter, mask, crop')
plt.show()


# Create an edge filter based on estimated core spacing, filter, crop and mask
imgCropped = pybundle.crop_rect(img, loc)[0]
edgeFilter = pybundle.edge_filter(np.shape(imgCropped)[0], coreSpacing * 1.8, coreSpacing * 0.1)

t1 = time.time()
imgProc = pybundle.filter_image(imgCropped, edgeFilter)
t2 = time.time()

print(f"Edge Filter Processing Time (ms): {round(1000 * (t2-t1),2)}")
plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title('Edge filter')
plt.show()
 



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
filterSize = 1.5

img = np.array(Image.open("data/bundle_colour_1.tif"))
calibImg = np.array(Image.open("data/bundle_colour_1_background.tif"))

# Locate the bundle
loc = pybundle.find_bundle(calibImg)

# Generate a mask by specifying the location of the bundle
mask = pybundle.get_mask(calibImg, loc)

# Estimate the core spacing
coreSpacing = pybundle.find_core_spacing(calibImg)

# Produce an image by Gaussian filtering, masking and cropping
t1 = time.time()
imgFilt = pybundle.g_filter(img, filterSize)
imgFilt = pybundle.apply_mask(imgFilt, mask)
imgFilt = pybundle.crop_rect(imgFilt, loc)[0]
t2 = time.time()
print(f"Gaussian Filter, Mask, Crop Processing Time (ms): { round(1000 * (t2-t1),2)}" )

# Produce an image using the simple Gaussian filtering, masking and cropping
t1 = time.time()
imgQuickFilt = pybundle.crop_filter_mask(img, loc, mask, filterSize)
t2 = time.time()
print(f"Gaussian Filter, Mask, Crop Processing Time (ms): { round(1000 * (t2-t1),2)}" )

# Create an edge filter based on estimated core spacing, filter, crop and mask
imgCropped = pybundle.crop_rect(img, loc)[0]
edgeFilter = pybundle.edge_filter(np.shape(imgCropped)[0], coreSpacing * 4, coreSpacing * 0.1)

t1 = time.time()
imgEdge = pybundle.apply_mask(img, mask)
imgEdge = pybundle.crop_rect(imgEdge, loc)[0]
imgEdge = pybundle.filter_image(imgEdge, edgeFilter)

t2 = time.time()
print(f"Edge Filter Processing Time (ms): {round(1000 * (t2-t1),2)}")


fig, axs = plt.subplots(2,2)
fig.suptitle('Test Simple Processing Colour')
axs[0,0].imshow(img, cmap='gray')

# set title of subplot
axs[0,0].title.set_text('Raw Image')

axs[1,0].imshow(imgFilt, cmap='gray')
axs[1,0].title.set_text('Combined G Filter, mask, crop')


axs[0,1].imshow(imgQuickFilt, cmap='gray')
axs[0,1].title.set_text('Sequential G Filter, mask, crop')

axs[1,1].imshow(imgEdge/256, cmap='gray')
axs[1,1].title.set_text('Edge filter')
plt.tight_layout()
plt.show()



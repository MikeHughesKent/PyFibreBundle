# -*- coding: utf-8 -*-
"""
Tests the find_cores function of pybundle.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import cv2 as cv

import context    # For paths to library

import pybundle


img = np.array(Image.open("data/bundle1.tif"))
imgMasked = pybundle.auto_mask(img)         # Remove anything outside bundle
cx,cy = pybundle.find_cores(imgMasked, 6)
plt.figure(dpi=600)
plt.imshow(imgMasked, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)
print(f"Found {str(len(cx))} cores.")


img = np.array(Image.open("data/bundle2.tif"))
img = img[:,:,0]
imgMasked = pybundle.auto_mask(img)
cx,cy = pybundle.find_cores(imgMasked, 8)
plt.figure(dpi=600)
plt.imshow(imgMasked, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)
print(f"Found {str(len(cx))} cores.")


img = np.array(Image.open("data/bundle3.tif"))
img = img[:,:,0]
imgMasked = pybundle.auto_mask(img)
cx,cy = pybundle.find_cores(imgMasked, 6)
plt.figure(dpi=600)
plt.imshow(imgMasked, cmap='gray')
plt.plot(cx,cy,'.', markersize = .1)
print(f"Found {str(len(cx))} cores.")



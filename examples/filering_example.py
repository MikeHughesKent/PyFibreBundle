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

import context

from pybundle import PyBundle

img = np.array(Image.open("../test/data/usaf1.tif"))
calibImg = np.array(Image.open("../test/data/usaf1_background.tif"))

# Create an instance of the PyBundle class 
pyb = PyBundle()

# Set to remove core pattern by Gaussian filtering
pyb.set_core_method(pyb.FILTER)
pyb.set_filter_size(2.5)

# Automatically create a mask using the calibration image
pyb.create_and_set_mask(calibImg)

# Set to crop to a square around bundle
pyb.set_crop(True)

# Do core removal, masking and cropping
imgProc = pyb.process(img)

plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
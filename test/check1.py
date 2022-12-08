# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 21:13:36 2022

@author: AOG
"""
import context

import numpy as np
  
import pybundle

from pybundle import PyBundle

from PIL import Image

import matplotlib.pyplot as plt

img = np.array(Image.open("data/usaf1.tif"))
calibImg = np.array(Image.open("data/usaf1_background.tif"))

pyb = PyBundle()
	
pyb.set_core_method(pyb.TRILIN)


pyb.set_calib_image(calibImg)
pyb.set_normalise_image(calibImg)


pyb.set_grid_size(512)

pyb.set_auto_contrast(True)

pyb.calibrate()


imgProc = pyb.process(img)

pyb.set_use_numba(False)

plt.imshow(imgProc)


coreSize = 3
gridSize = 512
calib = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, normalise = calibImg, automask = True)

imgProc = pybundle.recon_tri_interp(img, calib)

plt.imshow(imgProc)
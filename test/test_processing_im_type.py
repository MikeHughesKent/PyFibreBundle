# -*- coding: utf-8 -*-
"""
Testing that processing works with different image types.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
import time

from PIL import Image

from time import perf_counter as timer

import context

import pybundle 
from pybundle import PyBundle

filterSize = 2

img = np.array(Image.open("data/usaf1.tif"))
calibImg = np.array(Image.open("data/usaf1_background.tif"))


imTypes = ['uint8', 'uint16', 'float32', 'float64']
pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = filterSize, crop = True,  applyMask = True, calibImage = calibImg)
pyb.calibrate()
plt.figure()
for idx, imType in enumerate(imTypes):
    imgT = (img / 2).astype(imType)
    calibImage = img.astype(imType)
    t1 = timer()
    imgProc = pyb.process(imgT)
    print(f"Gaussian filter, {imType}: {round((timer() - t1) *1000)} ms.")
    plt.subplot(2, 2, idx + 1); plt.imshow(imgProc, cmap='gray');plt.title(imType)
plt.tight_layout()
    

imTypes = ['uint8', 'uint16', 'float32', 'float64']
pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = filterSize, crop = True,  applyMask = True, calibImage = calibImg, normaliseImage  = calibImg)
pyb.calibrate()
plt.figure()
for idx, imType in enumerate(imTypes):
    imgT = (img / 2).astype(imType)
    calibImage = img.astype(imType)
    t1 = timer()
    imgProc = pyb.process(imgT)
    print(f"Gaussian filter + Norm, {imType}: {round((timer() - t1) *1000)} ms.")
    plt.subplot(2, 2, idx + 1); plt.imshow(imgProc, cmap='gray');plt.title(imType)
plt.tight_layout()
    


imTypes = ['uint8', 'uint16', 'float32', 'float64']
pyb = PyBundle(coreMethod = PyBundle.EDGE_FILTER, edgeFilterShape = (6,1), crop = True,  applyMask = True, calibImage = calibImg)
pyb.calibrate()
plt.figure()
for idx, imType in enumerate(imTypes):
    imgT = img.astype(imType)
    calibImage = img.astype(imType)
    t1 = timer()
    imgProc = pyb.process(imgT)
    print(f"Edge filter, {imType}: {round((timer() - t1) *1000)} ms.")
    plt.subplot(2, 2, idx + 1); plt.imshow(imgProc, cmap='gray');plt.title(imType)
plt.tight_layout()


imTypes = ['uint8', 'uint16', 'float32', 'float64']
pyb = PyBundle(coreMethod = PyBundle.EDGE_FILTER, edgeFilterShape = (6,1), crop = True,  applyMask = True, calibImage = calibImg, normaliseImage  = calibImg)
pyb.calibrate()
plt.figure()
for idx, imType in enumerate(imTypes):
    imgT = img.astype(imType)
    t1 = timer()
    imgProc = pyb.process(imgT)
    print(f"Edge filter + Norm, {imType}: {round((timer() - t1) *1000)} ms.")
    plt.subplot(2, 2, idx + 1); plt.imshow(imgProc, cmap='gray');plt.title(imType)
plt.tight_layout()



imTypes = ['uint8', 'uint16', 'float32', 'float64']
pyb = PyBundle(coreMethod = PyBundle.TRILIN, calibImage = calibImg)
pyb.calibrate()
plt.figure()
for idx, imType in enumerate(imTypes):
    imgT = img.astype(imType)
    imgProc = pyb.process(imgT)   # Warm up numpy for this data type
    t1 = timer()
    imgProc = pyb.process(imgT)
    print(f"Linear interp, {imType}: {round((timer() - t1) *1000)} ms.")
    plt.subplot(2, 2, idx + 1); plt.imshow(imgProc, cmap='gray');plt.title(imType)   
plt.tight_layout()
   
    

imTypes = ['uint8', 'uint16', 'float32', 'float64']
pyb = PyBundle(coreMethod = PyBundle.TRILIN, calibImage = calibImg, normaliseImage  = calibImg)
pyb.calibrate()
plt.figure()
for idx, imType in enumerate(imTypes):
    imgT = img.astype(imType)
    imgProc = pyb.process(imgT)   # Warm up numpy for this data type
    t1 = timer()
    imgProc = pyb.process(imgT)
    print(f"Linear interp + Norm, {imType}: {round((timer() - t1) *1000)} ms.")
    plt.subplot(2, 2, idx + 1); plt.imshow(imgProc, cmap='gray');plt.title(imType)        #
plt.tight_layout()
    
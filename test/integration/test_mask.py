"""
Tests the masking functionality of PyBundle class.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
import numpy as np

from PIL import Image


from pybundle import PyBundle 

from pathlib import Path


img = np.array(Image.open(Path('data/usaf1.tif')))
calibImg = np.array(Image.open(Path('data/usaf1_background.tif')))



pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2)
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('Default'); plt.imshow(imgProc[:5,:5], cmap = 'gray')

assert imgProc[0,0] != 0



pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, applyMask = True)
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('applyMask = True'); plt.imshow(imgProc[:5,:5], cmap = 'gray')

assert imgProc[0,0] == 0



pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, applyMask = True, autoLoc = False)
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('applyMask = True'); plt.imshow(imgProc[:5,:5], cmap = 'gray')

assert imgProc[0,0] != 0



pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, calibImage = calibImg, applyMask = True)
pyb.calibrate()
assert pyb.mask is not None
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('applyMask = True'); plt.imshow(imgProc[:5,:5], cmap = 'gray')

assert imgProc[0,0] == 0

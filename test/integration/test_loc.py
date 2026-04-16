# -*- coding: utf-8 -*-
"""
Tests the location find routines

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
import numpy as np
from pathlib import Path

from PIL import Image


from pybundle import PyBundle 


img = np.array(Image.open(Path('data/usaf1.tif')))
calibImg = np.array(Image.open(Path('data/usaf1_background.tif')))


pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2)
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('Default'); plt.imshow(imgProc, cmap = 'gray')

assert np.shape(imgProc)[0] != np.shape(imgProc)[1] 



pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, crop = True)
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('crop = True'); plt.imshow(imgProc, cmap = 'gray')

assert np.shape(imgProc)[0] == np.shape(imgProc)[1] 




pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, crop = True, autoLoc = False)
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('crop = True, autoLoc = False'); plt.imshow(imgProc, cmap = 'gray')

assert np.shape(imgProc)[0] != np.shape(imgProc)[1] 




pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, crop = True, autoLoc = False, loc = (200,200,100) )
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('crop = True, autoLoc default loc set'); plt.imshow(imgProc, cmap = 'gray')

assert np.shape(imgProc) == (200,200)





pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, crop = True, autoLoc = True, loc = (200,200,100) )
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('crop = True, autoLoc = True, loc set'); plt.imshow(imgProc, cmap = 'gray')

assert np.shape(imgProc) != (200,200)
assert np.shape(imgProc)[0] == np.shape(imgProc)[1] 




pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, 
               crop = True, calibImage = calibImg )
pyb.calibrate()
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('crop = True, autoLoc = Default, calibrate'); plt.imshow(imgProc, cmap = 'gray')

assert np.shape(imgProc)[0] == np.shape(imgProc)[1] 





pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, 
               crop = True, autoLoc = False, calibImage = calibImg )
pyb.calibrate()
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('crop = True, autoLoc = False, calibrate'); plt.imshow(imgProc, cmap = 'gray')

assert np.shape(imgProc)[0] != np.shape(imgProc)[1] 





pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2, 
               crop = True, loc = (200,200,100), calibImage = calibImg)
pyb.calibrate()
imgProc = pyb.process(img)
#plt.figure(dpi = 150); plt.title('crop = True, loc set, calibrate'); plt.imshow(imgProc, cmap = 'gray')

assert np.shape(imgProc) == (200,200)

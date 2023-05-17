# -*- coding: utf-8 -*-
"""
Simple example of how to use PyFibreBundle to remove core pattern with an edge filter.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from time import perf_counter as timer

from pathlib import Path

import context

from pybundle import PyBundle

# Load images
img = np.array(Image.open(Path('../test/data/usaf1.tif')))
calibImg = np.array(Image.open(Path('../test/data/usaf1_background.tif')))

# Create an instance of the PyBundle class, set to remove core pattern by Edge filtering
pyb = PyBundle(coreMethod = PyBundle.EDGE_FILTER, 
               edgeFilterShape = (10,2) ) 
               
#pyb.calibrate()
t1 = timer()
imgProc = pyb.process(img)
print(f"Calibration and Edge filter took {round((timer() - t1) * 1000)} ms ")



# Create an instance of the PyBundle class, set to remove core pattern by Edge filtering using
# calibration image to pre-determine the size of bundle and pre-calculating the filter
pyb = PyBundle(coreMethod = PyBundle.EDGE_FILTER, 
               edgeFilterShape = (10,2), calibImage = calibImg) 
               
pyb.calibrate()
t1 = timer()
imgProc = pyb.process(img)
print(f"Edge filter took {round((timer() - t1) * 1000)} ms ")


plt.figure(dpi=300)
plt.imshow(imgProc, cmap='gray')
plt.title("Edge filter")
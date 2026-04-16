# -*- coding: utf-8 -*-
"""
Tests the super-resolution stack sorter helper function.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import context
import pybundle as pyb
import os
import numpy as np
from PIL import Image

folder = "data\super_res_2\data"
numIms = 6
stackSize = 4

files = [f.path for f in os.scandir(folder)]

meanVal = np.zeros(len(files))

img = Image.open(files[0])

(w,h) = np.shape(img)

imgs = np.zeros((w,h,numIms))
for idx in range(numIms):
    imgs[:,:,idx] = Image.open(files[idx])
    

sortedStack = pyb.SuperRes.sort_sr_stack(imgs, stackSize)
# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.


This file contains utility functions.


@author: Mike Hughes
Applied Optics Group, University of Kent
https://github.com/mikehugheskent
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time

import cv2 as cv



# Extract a central square from an image. The extracted square is centred
# on the input image, with size 2 * boxSize if possible.
def extract_central(img, boxSize):
    w = np.shape(img)[0]
    h = np.shape(img)[1]

    cx = w/2
    cy = h/2
    boxSemiSize = min(cx,cy,boxSize)
    
    imgOut = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
    return imgOut





# Returns an 8 bit representation of image. If min and max are specified,
# these pixel values in the original image are mapped to 0 and 255 
# respectively, otherwise the smallest and largest values in the 
# whole image are mapped to 0 and 255, respectively.
def to8bit(img, **kwargs):
    minV = kwargs.get("minVal", None)
    maxV = kwargs.get("maxVal", None)
        
        
    img = img.astype('float64')
       
    if minV is None:
        minV = np.min(img)
            
                    
    img = img - minV
        
    if maxV is None:
        maxV = np.max(img)
    else:
        maxV = maxV - minV
        
    img = img / maxV * 255
    img = img.astype('uint8')
    return img





# Produce angular averaged radial profile through image img centred on
# centre, a tuple of (x_centre, y_centre).
def radial_profile(img, centre):
    y, x = np.indices((img.shape))
    r = np.sqrt((x - centre[1])**2 + (y - centre[0])**2)
    r = r.astype(np.int)

    
    tbin = np.bincount(r.ravel(), weights = img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin  / nr
    
    return radialprofile 
     
     
 

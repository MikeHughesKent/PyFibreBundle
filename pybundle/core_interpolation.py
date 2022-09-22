# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.


This file contains functions related to triangular linear interpolation 
between cores, including functions for finding cores.


@author: Mike Hughes
Applied Optics Group, University of Kent
https://github.com/mikehugheskent
"""


import numpy as np
import math
import matplotlib.pyplot as plt
import time

import cv2 as cv

from scipy.spatial import Delaunay

import pybundle
from pybundle.bundle_calibration import BundleCalibration

from numba import jit

import numba

 # Find cores in bundle image using regional maxima. Generally fast and
 # accurate. CoreSpacing is an estimate of the separation between cores in
 # pixels

def find_cores(img, coreSpacing):

     # Pre-filtering helps to minimse noise and reduce efffect of
     # multimodal patterns
     imgF = pybundle.g_filter(img, coreSpacing/5)

     imgF = (imgF / np.max(imgF) * 255).astype('uint8')

     # Find regional maximum by taking difference between dilated and original
     # image. Because of the way dilation works, the local maxima are not changed
     # and so these will have a value of 0
     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (coreSpacing,coreSpacing))
     imgD = cv.dilate(imgF, kernel)
     imgMax = 255 - (imgF - imgD)  # we need to invert the image

     # Just keep the maxima
     thres, imgBinary = cv.threshold(imgMax, 0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)

     # Dilation step helps deal with mode patterns which have led to multiple
     # maxima within a core, the two maxima will end up merged into one connected
     # region
     elSize = math.ceil(coreSpacing / 3)
     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (elSize,elSize))
     imgDil = cv.dilate(imgBinary, kernel)

     # Core centres are centroids of connected regions
     nReg, p1, p2, centroid = cv.connectedComponentsWithStats(
         imgDil, 8, cv.CV_32S)

     regSizes = p2[:, 4]   # Sizes of the regions. Used to reject backgrund regions

     cx = centroid[regSizes < coreSpacing**2, 0]  
     cy = centroid[regSizes < coreSpacing**2, 1]
     # print(np.shape(cx))
     # print(np.shape(cy))
     # # Method can find cores with centres outside of image. This is
     # # unwanted as tends to lead to indexing errors in other parts of code, so
     # # remove them here
     cx = cx[cx < np.shape(img)[1]]
     cy = cy[cx < np.shape(img)[1]]

     cx = cx[cx >= 0]
     cy = cy[cx >= 0]

     cy = cy[cy < np.shape(img)[0]]
     cx = cx[cy < np.shape(img)[0]]

     cy = cy[cy >= 0]
     cx = cx[cy >= 0]


     return cx, cy






 # Extract intensity of each core in fibre bundle image. First applies a
 # Gaussian filter of sigma 'filterSize' unless filterSize = 0. 'cx' and 'cy'
 # are vectors containing the x and y centres of each core.

def core_values(img, coreX, coreY, filterSize, **kwargs):
    

     numba = kwargs.get('numba', False)
     
     if filterSize is not None:
         img = pybundle.g_filter(img, filterSize)

     if numba:
         cInt = core_value_extract_numba(img, coreX, coreY)
     else:
         cInt = img[coreY, coreX]
        
     return cInt
 
    
# Faster core value extraction if numba is being used 
@jit(nopython = True)    
def core_value_extract_numba(img, coreX, coreY):
    
    #img2 = img.astype('int')
    nCores = np.shape(coreX)[0]
    cInt = np.zeros((nCores),dtype=numba.int64)
    for i in range(nCores):
        cInt[i] = (img[coreY[i], coreX[i]])
    return cInt

 # Performs calibration to allow subsequent core removal by triangular linear interpolation.
 # Reconstructed images will be of size (gridSize, gridSize). 'coreSize' is used by
 # the core finding routine, and should be an estimate of the core spacing. 'centreX' and
 # 'centreY' are the positions in the original image that the reconstruction will be centred
 # on.
 # Thanks to Cheng Yong Xin, Joseph, who collaborated in implementation of this function.

def calib_tri_interp(img, coreSize, gridSize, **kwargs):

     centreX = kwargs.get('centreX', -1)
     centreY = kwargs.get('centreY', -1)
     radius = kwargs.get('radius', -1)
     filterSize = kwargs.get('filterSize', 0)
     normalise = kwargs.get('normalise', None)
     autoMask = kwargs.get('autoMask', True)
     mask = kwargs.get('mask', True)
     background = kwargs.get('background', None)

     if autoMask:
         img = pybundle.auto_mask(img)

     # Find the cores in the calibration image
     coreX, coreY = pybundle.find_cores(img, coreSize)
     
     coreX = np.round(coreX).astype('uint16')
     coreY = np.round(coreY).astype('uint16')


     # Default values
     if centreX < 0:
         centreX = np.mean(coreX)
     if centreY < 0:
         centreY = np.mean(coreY)
     if radius < 0:
         dist = np.sqrt((coreX - centreX)**2 + (coreY - centreY)**2)
         radius = max(dist)

     # Delaunay triangulation and find barycentric co-ordinates for each pixel
     calib = pybundle.init_tri_interp(img, coreX, coreY, centreX, centreY, radius, gridSize, filterSize= filterSize, background = background, normalise = normalise, mask = mask)

     return calib


 # Performs Delaunay triangulation of core positions, and finds each pixel of
 # reconstruction grid in barycentric co-ordinates w.r.t. enclosing triangle
def init_tri_interp(img, coreX, coreY, centreX, centreY, radius, gridSize, **kwargs):

     filterSize = kwargs.get('filterSize', 0)
     normalise = kwargs.get('normalise', None)
     background = kwargs.get('background', None)
     mask = kwargs.get('mask', True)

     # Delaunay triangulation over core centre locations
     points = np.vstack((coreX, coreY)).T
     tri = Delaunay(points)

     # Make a vector of all the pixels in the reconstruction grid
     xPoints = np.linspace(centreX - radius, centreX + radius, gridSize)
     yPoints = np.linspace(centreY - radius, centreY + radius, gridSize)
     nPixels = gridSize**2
     mX, mY = np.meshgrid(xPoints, yPoints)
     interpPoints = np.vstack( ( np.reshape(mX, nPixels) , np.reshape(mY,nPixels) )).T

     # Find enclosing triangle for each pixel in reconstruction grid
     t1 = time.perf_counter()
     mapping = tri.find_simplex(interpPoints, bruteforce=False, tol=None)
     #print(time.perf_counter() - t1)

     # Write each pixel position in terms of barycentric co-ordinates w.r.t
     # enclosing triangle.
     baryCoords = np.zeros([nPixels, 3])
     for i in range(nPixels):
         b = tri.transform[mapping[i], :2].dot(np.transpose(interpPoints[i,:] - tri.transform[mapping[i],2]))
         baryCoords[i, 0:2] = b
         baryCoords[i, 2] = 1 - b.sum(axis=0)   # Third co-ordinate found from
                                               # making sure sum of all is 1

     # Store background values
     if normalise is not None:
         normaliseVals = pybundle.core_values(normalise, coreX, coreY,filterSize).astype('double')
     else:
         normaliseVals = 0


     if background is not None:
         backgroundVals = pybundle.core_values(background, coreX, coreY,filterSize).astype('double')
     else:
         backgroundVals = 0

     calib = BundleCalibration()

     calib.radius = radius
     calib.coreX = coreX
     calib.coreY = coreY
     calib.gridSize = gridSize
     calib.tri = tri
     calib.filterSize = filterSize
     calib.baryCoords = baryCoords
     calib.mapping = mapping
     calib.normalise = normalise
     calib.normaliseVals = normaliseVals
     calib.background = background
     calib.backgroundVals = backgroundVals

     calib.coreIdx = calib.tri.vertices[calib.mapping, :]

     if mask:
         calib.mask = pybundle.get_mask(np.zeros((gridSize, gridSize)), (gridSize/2, gridSize/2,gridSize/2))
     else:
         calib.mask = None

     return calib

 # Get core normalisation values from calibration image

def tri_interp_normalise(calibIn, normalise):
     calibOut = calibIn

     if normalise is not None:
         calibOut.normaliseVals = pybundle.core_values(normalise, calibOut.coreX, calibOut.coreY,calibOut.filterSize).astype('double')
         calibOut.normalise = normalise

     else:
         calibOut.normaliseVals = 0
         calibOut.normalise = None
     return calibOut


 # Get core background values from calibration image

def tri_interp_background(calibIn, background) :
     calibOut = calibIn

     if background is not None:
         calibOut.backgroundVals = pybundle.core_values(background, calibOut.coreX, calibOut.coreY,calibOut.filterSize).astype('double')
         calibOut.background = background

     else:
         calibOut.backgroundVals = 0
         calibOut.background = None
     return calibOut

 # Removes core pattern using triangular linear interpolation. Requires an initial
 # calibration using calibTriInterp

def recon_tri_interp(img, calib, **kwargs):
    
     numba = kwargs.get('numba', False)

     # Extract intensity from each core
     t1 = time.perf_counter()

     cVals = pybundle.core_values(
         img, calib.coreX, calib.coreY, calib.filterSize, **kwargs).astype('double')
     #print("Core values took:", time.perf_counter() - t1)    

     if calib.background is not None:
         cVals = cVals - calib.backgroundVals

     t1 = time.perf_counter()
      
     if calib.normalise is not None:
         cVals = (cVals / calib.normaliseVals * 255) 
     #print("Normalisation took:", time.perf_counter() - t1)    
   

     # Triangular linear interpolation
    # pixelVal = np.zeros_like(calib.mapping, dtype='uint8')  

     t1 = time.perf_counter()
     if numba:
         if calib.mask is not None:
             maskNumba = np.squeeze(np.reshape(calib.mask, (np.product(np.shape(calib.mask)),1)))
         else:
             maskNumba = None
         pixelVal = grid_data_numba(calib.baryCoords, cVals, calib.coreIdx, calib.mapping, maskNumba)
     else:
         pixelVal = grid_data(calib.baryCoords, cVals, calib.coreIdx, calib.mapping)
     #print("Grid data took:", time.perf_counter() - t1)    

     # Vector of pixels now has to be converted to a 2D image
     
     t1 = time.perf_counter()
     pixelVal = np.reshape(pixelVal, (calib.gridSize, calib.gridSize))
     #print("Reshape took:", time.perf_counter() - t1)    


     t1 = time.perf_counter()
     if calib.mask is not None:
         if numba:
             pass
             #pixelVal = apply_mask_numba(pixelVal, calib.mask)
         else:    
             pixelVal = pybundle.apply_mask(pixelVal, calib.mask)
     #print("Masking took:", time.perf_counter() - t1)    


     return pixelVal
 

# Produces pixel values based on linear interpolation between supplied
# surrounding triangle vertex values (cVals) using pre-calculated
# barycentric co-ordinates (baryCoords). Mapping contains the triangle
# id for each pixel, if this is -1 then the pixel is outisde the complex 
# hull and so is set to zero. barycoords should be of size (3, num_pixels). 
# coreIdx stores the core id for each of the three cores surrounding each pixel,
# and is of size (3, num_pixels).
def grid_data(baryCoords, cVals, coreIdx, mapping):
    
     val = baryCoords * cVals[coreIdx]
     pixelVal = np.sum(val, 1)
     pixelVal[mapping < 0] = 0
     
     return pixelVal


# Numba optimsied version of grid_data. 
# Produces pixel values based on linear interpolation between supplied
# surrounding triangle vertex values (cVals) using pre-calculated
# barycentric co-ordinates (baryCoords). Mapping contains the triangle
# id for each pixel, if this is -1 then the pixel is outisde the complex 
# hull and so is set to zero. barycoords should be of size (3, num_pixels). 
# coreIdx stores the core id for each of the three cores surrounding each pixel,
# and is of size (3, num_pixels).
@jit(nopython=True)
def grid_data_numba(baryCoords, cVal, coreIdx, mapping, mask):

    pixelVal = np.zeros((np.shape(baryCoords)[0]))
    if mask is None:

        for i in range(np.shape(baryCoords)[0]):
            pixelVal[i] = baryCoords[i,0] * cVal[coreIdx[i,0]] + baryCoords[i,1] * cVal[coreIdx[i,1]]+ baryCoords[i,2] * cVal[coreIdx[i,2]]
    else: 
       # print("is a mask")
        for i in range(np.shape(baryCoords)[0]):
           # print(baryCoords[i,0])
            #print(mask[i,0])
            pixelVal[i] = int(mask[i]) * ( baryCoords[i,0] * cVal[coreIdx[i,0]] + baryCoords[i,1] * cVal[coreIdx[i,1]]+ baryCoords[i,2] * cVal[coreIdx[i,2]])
        
    
    pixelVal[mapping < 0] = 0
   
    return pixelVal   
        
@jit(nopython=True, fastmath=True)
def apply_mask_numba(img, mask):
    
    w,h = np.shape(img)
   
    for x in range(w):
        for y in range(h):
            img[x,y] = img[x,y] * mask[x,y]
           
    return img       
    
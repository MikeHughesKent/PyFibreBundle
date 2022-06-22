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
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
from scipy.spatial import Delaunay

import pybundle
from pybundle.bundle_calibration import BundleCalibration


# Find cores in bundle image using Hough transform. This generally
# does not work as well as findCores and is a lot slower!
def find_cores_hough(img, **kwargs):

     scaleFac = kwargs.get('scaleFactor', 2)
     cannyLow = kwargs.get('cannyLow', .05)
     cannyHigh = kwargs.get('cannyHigh', .8)
     estRad = kwargs.get('estRad', 1)
     minRad = kwargs.get('minRad', np.floor(max(1, estRad)).astype('int'))
     maxRad = kwargs.get('maxRad', np.floor(minRad + 2).astype('int'))
     minSep = kwargs.get('minSep', estRad * 2)
     darkRemove = kwargs.get('darkRemove', 2)
     gFilterSize = kwargs.get('filterSize', estRad / 2)

     imgR = cv.resize(img, [scaleFac * np.size(img, 1),
                      scaleFac * np.size(img, 0)]).astype(float)

     # Pre filter with Gaussian and Canny
     imgF = pybundle.g_filter(imgR, gFilterSize*scaleFac)
     imgF = imgF.astype('uint8')
     edges = cv.Canny(imgF, cannyLow, cannyHigh)

     # Using Scikit-Image Hough implementation, trouble getting CV to work
     radii = range(math.floor(minRad * scaleFac),
                   math.ceil(maxRad * scaleFac))
     circs = hough_circle(edges, radii, normalize=True, full_output=False)

     minSepScaled = np.round(minSep * scaleFac).astype('int')

     for i in range(np.size(circs, 0)):
         circs[i, :, :] = np.multiply(circs[i, :, :], imgF)

     accums, cx, cy, radii = hough_circle_peaks(
         circs, radii, min_xdistance=minSepScaled, min_ydistance=minSepScaled)

     # Remove any finds that lie on dark points
     meanVal = np.mean(imgF[cy, cx])
     stdVal = np.std(imgF[cy, cx])
     removeCore = np.zeros_like(cx)
     for i in range(np.size(cx)):
         if imgF[cy[i], cx[i]] < meanVal - darkRemove * stdVal:
             removeCore[i] = 1
     cx = cx[removeCore != 1]
     cy = cy[removeCore != 1]

     cx = cx / scaleFac
     cy = cy / scaleFac

     return cx, cy

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

def core_values(img, coreX, coreY, filterSize):


     cxR = np.round(coreX).astype('int16')
     cyR = np.round(coreY).astype('int16')

     if filterSize is not None:
         img = pybundle.g_filter(img, filterSize)

     cInt = img[cyR, cxR]

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

def recon_tri_interp(img, calib):

     # Extract intensity from each core
     cVals = pybundle.core_values(
         img, calib.coreX, calib.coreY, calib.filterSize).astype('double')
     if calib.background is not None:
         cVals = cVals - calib.backgroundVals

     if calib.normalise is not None:
         cVals = cVals / calib.normaliseVals * 255

     # Triangular linear interpolation
     pixelVal = np.zeros_like(calib.mapping, dtype='double')  

     val = calib.baryCoords * cVals[calib.coreIdx]
     pixelVal = np.sum(val, 1)
     pixelVal[calib.mapping < 0] = 0


     # Vector of pixels now has to be converted to a 2D image
     pixelVal = np.reshape(pixelVal, (calib.gridSize, calib.gridSize))

     if calib.mask is not None:
         pixelVal = pybundle.apply_mask(pixelVal, calib.mask)

     return pixelVal

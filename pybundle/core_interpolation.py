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


# We try to import numba here and if successful, load the numba-optimised
# interpolation funactions. If we get an error (i.e. library not available)
# then we won't call the function that require this.
try:
    from numba import jit
    import numba
    from pybundle.core_interpolation_numba import *
    numbaAvailable = True
except:
    numbaAvailable = False
 

def find_cores(img, coreSpacing):
     """ Find cores in bundle image using regional maxima. Generally fast and
     accurate. CoreSpacing is an estimate of the separation between cores in
     pixels.
     """
     
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


def core_values(img, coreX, coreY, filterSize, **kwargs):
    """ Extract intensity of each core in fibre bundle image. First applies a
    Gaussian filter unless filterSize is None. Supports JIT acceleration
    if numba is installed.
    :param coreX: 1D numpy array giving x co-ordinates of core centres
    :param coreY: 1D numpy array giving y co-ordinates of core centres
    :param filterSize: sigma of Gaussian filter
    :param numba: optional, if true numba JIT used for faster executio, defaults to False.
    """
    numba = kwargs.get('numba', False)
     
    if filterSize is not None:
        img = pybundle.g_filter(img, filterSize)

    if numba:
        cInt = core_value_extract_numba(img, coreX, coreY)
    else:
        cInt = img[coreY, coreX]
        
    return cInt
 
    
def calib_tri_interp(img, coreSize, gridSize, **kwargs):
    """ Performs calibration to allow subsequent core removal by triangular linear interpolation.
    Reconstructed images will be of size (gridSize, gridSize). 'coreSize' is used by
    the core finding routine, and should be an estimate of the core spacing. This function
    returns the entire calibration as an instance of BundleCalibration which can subsequently by used 
    by recon_tri_interp. If background and/or normalisation images are specified, subsequent 
    reconstructions will have background subtraction and/or normalisation respectively.
        
    Thanks to Cheng Yong Xin, Joseph, who collaborated in implementation of this function.
    
    :param img: calibration image of bundle as 2D numpy array
    :param coreSize: estimate of average spacing between cores
    :param gridSize: output size of image, supply a single value, image will be square
    :param centreX: optional, x centre location of bundle, if not specified will be determined automatically
    :param centreY: optional, y centre location of bundle, if not specified will be determined automatically
    :param radius: optional, radius of bundle, if not specified will be determined automatically
    :param filterSize: optional, sigma of Gaussian filter applied, defaults to  0 (no filter)
    :param background: optional, image used for background subtractionn as 2D numpy array
    :param normalise: optional, image used for normalisation, as 2D numpy array. Can be same as 
                      calibration image, defaults to no normalisation
    :param autoMask: optional, boolean, if true the calibration image will be masked to prevent 
                     spurious core detections outside of bundle, defualts to True
    :param mask: optional, boolean, when reconstructing output image will be masked outside of 
                 bundle, defaults to True
    :return: instance of BundleCalibration
    """

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

 
def init_tri_interp(img, coreX, coreY, centreX, centreY, radius, gridSize, **kwargs):
    """ Used by calib_tri_interp to perform Delaunay triangulation of core positions, 
    and find each pixel of reconstruction grid in barycentric co-ordinates w.r.t. 
    enclosing triangle.
    :param img: calibration image as 2D numpy array
    :param coreX: x centre of each core as 1D numpy array
    :param coreY: y centre of each core as 1D numpy array
    :param centreX: x centre location of bundle (reconstruction will be centred on this)
    :param centreY: y centre location of bundle (reconstruction will be centred on this)
    :param radius: radius of bundle (reconstruction will cover a square out to this radius)
    :param gridSize: output size of image, supply a single value, image will be square
    :param filterSize: optional, sigma of Gaussian filter applied, defaults to no filter
    :param background: optional, image used for background subtractionn as 2D numpy array
    :param normalise: optional, image used for normalisation, as 2D numpy array. Can be same as 
                      calibration image, defaults to no normalisation
    :param mask: optional, boolean, when reconstructing output image will be masked outside 
                 of bundle, defaults to True
    :return: instance of BundleCalibration
    
    """
    filterSize = kwargs.get('filterSize', 0)
    normalise = kwargs.get('normalise', None)
    background = kwargs.get('background', None)
    mask = kwargs.get('mask', True)
    numba = kwargs.get('numba', True)


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
    mapping = tri.find_simplex(interpPoints, bruteforce=False, tol=None)

    # Write each pixel position in terms of barycentric co-ordinates w.r.t
    # enclosing triangle.
    baryCoords = barycentric(nPixels, tri, mapping, interpPoints)
   
    # Store background values
    if normalise is not None:
        normaliseVals = pybundle.core_values(normalise, coreX, coreY,filterSize).astype('double')
    else:
        normaliseVals = 0

    if background is not None:
        backgroundVals = pybundle.core_values(background, coreX, coreY,filterSize).astype('double')
    else:
        backgroundVals = 0

    # The calibration is stored in in instance of BundleCalibration
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


def barycentric(nPixels, tri, mapping, interpPoints):
    """ Converts a set of Cartesian co-ordinates to barycentric co-ordinates.
    Assumes a prior Delaunay triangulation and simplex stored in mapping.
    :param nPixels : Number of pixels in grid 
    :param tri     : Delaunay triangulation
    :param mapping : Output from Delaunay.find_simplex, records the co-ordinates of the
                     three triangle vertices surrounding each point
    :param interpPoints : Cartesian grid co-ordinates
    """
    baryCoords = np.zeros([nPixels, 3])
    for i in range(nPixels):
        c = tri.transform[mapping[i]]
        b = c[:2].dot(np.transpose(interpPoints[i,:] - c[2]))
        baryCoords[i, 0:2] = b
        baryCoords[i, 2] = 1 - b.sum(axis=0)  # Third co-ordinate found from
                                              # making sure sum of all is 1
                                              # Third co-ordinate found from
    return baryCoords


def tri_interp_normalise(calibIn, normalise):
    """ Updates a calibration with a new normalisation without requiring
    full recalibration
    :param calibIn: input calibration, instance of BundleCalibration
    :param normalise: normalisation image as 2D numpy array, or None to remove normalisation
    :return: updated instance of BundleCalibration
    """
    calibOut = calibIn

    if normalise is not None:
        calibOut.normaliseVals = pybundle.core_values(normalise, calibOut.coreX, calibOut.coreY,calibOut.filterSize).astype('double')
        calibOut.normalise = normalise

    else:
        calibOut.normaliseVals = 0
        calibOut.normalise = None
    
    return calibOut


def tri_interp_background(calibIn, background) :
     """ Updates a calibration with a new background without requiring
     full recalibration
     :param calibIn: bundle calibration, instance of BundleCalibration
     :param background: background image as 2D numpy array
     """
     calibOut = calibIn

     if background is not None:
         calibOut.backgroundVals = pybundle.core_values(background, calibOut.coreX, calibOut.coreY,calibOut.filterSize).astype('double')
         calibOut.background = background

     else:
         calibOut.backgroundVals = 0
         calibOut.background = None
         
     return calibOut


def recon_tri_interp(img, calib, **kwargs):
     """ Removes core pattern using triangular linear interpolation. Requires an initial
     calibration using calib_tri_interp
     :param img: raw image to be reconstructed as 2D numpy array
     :param calib: bundle calibration as instance of BundleCalibration
     :param numba: optional, if true use JIT acceleration using Numba, default to False
     :return: reconstructed image as 2D numpy array
     """
    
     numba = kwargs.get('numba', True)

     # Extract intensity from each core
     cVals = pybundle.core_values(
         img, calib.coreX, calib.coreY, calib.filterSize, **kwargs).astype('double')

     if calib.background is not None:
         cVals = cVals - calib.backgroundVals

      
     if calib.normalise is not None:
         cVals = (cVals / calib.normaliseVals * 255) 
   

     # Triangular linear interpolation
     if numba and numbaAvailable:
         if calib.mask is not None:
             maskNumba = np.squeeze(np.reshape(calib.mask, (np.product(np.shape(calib.mask)),1)))
         else:
             maskNumba = None
         pixelVal = grid_data_numba(calib.baryCoords, cVals, calib.coreIdx, calib.mapping, maskNumba)
     else:
         pixelVal = grid_data(calib.baryCoords, cVals, calib.coreIdx, calib.mapping)

     # Vector of pixels now has to be converted to a 2D image
     pixelVal = np.reshape(pixelVal, (calib.gridSize, calib.gridSize))

     if calib.mask is not None:
         if numba and numbaAvailable:
             pass   # masking is done in one step earlier on
         else:    
             pixelVal = pybundle.apply_mask(pixelVal, calib.mask)

     return pixelVal
 

def grid_data(baryCoords, cVals, coreIdx, mapping):
    """Produces pixel values based on linear interpolation between supplied
    surrounding triangle vertex core intensity values. The core values, cVals, 
    are supplied as a 1D numpy array, giving the intensity extraacted from each core.
    The indices of the three cores surrounding each pixel in the reconstruction
    grid are given by coreIdx, a 2D  numpy array. The mapping, which provides
    the tringle index for each reconstruction grid pixel, is used here purely
    for intentifying pixels which lie outside the complex hull of the bundle, so
    that these can be set to 0.  
    :param baryCoords: barycentric co-ordinates of each pixel in reconstruction grid
        as 2D numpy array of size (3, num_pixels)
    :param cVals: intensity values from each core as 1D numpy array
    :param coreIdx: the indices of the three cores making up the enclosing triangle
        for each reconstruction grid pixel, as 2D numpy array of size (3, num_pixels)
    :return: value of each pixel in the reconstruction grid, as 1D numpy array
    """
    
    val = baryCoords * cVals[coreIdx]
    pixelVal = np.sum(val, 1)
    pixelVal[mapping < 0] = 0
     
    return pixelVal
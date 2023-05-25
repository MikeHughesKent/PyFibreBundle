# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains functions related to triangular linear interpolation 
between cores, including functions for finding cores.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""


import numpy as np
import math
import time


import matplotlib.pyplot as plt

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
     accurate. 
     
     Returns tuple of (x_pos, y_pos) where x_pos and y_pos are 1D numpy arrays.
     
     Arguments:
         img         : 2D/3D numpy array
         coreSpacing : float, estimate of the separation between cores in
                       pixels.
     """
     # Pre-filtering helps to minimse noise and reduce efffect of
     # multimodal patterns
     imgF = pybundle.g_filter(img.astype('float32'), coreSpacing/5)

     # If a colour image, convert to greyscale by taking the maximum value across the channels
     imgF = pybundle.max_channels(imgF)

     imgF = (imgF / np.max(imgF) * 255).astype('uint8') 

     # Find regional maximum by taking difference between dilated and original
     # image. Because of the way dilation works, the local maxima are not changed
     # and so these will have a value of 0
     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (int(round(coreSpacing)),int(round(coreSpacing)) ))
     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3 ))

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
     
     # Method can find cores with centres outside of image. This is
     # unwanted as tends to lead to indexing errors in other parts of code, so
     # remove them here
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
    
    Arguments:
        coreX      : 1D numpy array giving x co-ordinates of core centres
        coreY      : 1D numpy array giving y co-ordinates of core centres
        filterSize : float, sigma of Gaussian filter
    
    Keyword Arguments:    

        numba  : optional, if true numba JIT used for faster execution, defaults to False.
    """
    numba = kwargs.get('numba', False)
    
    if filterSize is not None:
        img = pybundle.g_filter(img, filterSize)

    cInt = np.zeros(np.shape(coreX))
    if numba and numbaAvailable:     
        cInt = cInt + core_value_extract_numba(img, coreX, coreY)         
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
    
    Returns an instance of BundleCalibration

        
    Thanks to Cheng Yong Xin, Joseph, who collaborated in implementation of this function.
    
    Arguments:
        img        : calibration image of bundle as 2D (mono) or 3D (colour) numpy array
        coreSize   : float, estimate of average spacing between cores
        gridSize   : int, output size of image, supply a single value, image will be square
        
    Keyword Arguments:    
        centreX    : int, optional, x centre location of bundle, if not specified will be determined automatically
        centreY    : int, optional, y centre location of bundle, if not specified will be determined automatically
        radius     : int, optional, radius of bundle, if not specified will be determined automatically
        filterSize : float, optional, sigma of Gaussian filter applied, defaults to  0 (no filter)
        background : optional, image used for background subtraction as 2D numpy array
        normalise  : optional, image used for normalisation, as 2D numpy array. Can be same as 
                     calibration image, defaults to no normalisation
        autoMask   : optional, boolean, if true the calibration image will be masked to prevent 
                     spurious core detections outside of bundle, defualts to True
        mask       : optional, boolean, when reconstructing output image will be masked outside of 
                     bundle, defaults to True
        
    """

    centreX = kwargs.get('centreX', None)
    centreY = kwargs.get('centreY', None)
    radius = kwargs.get('radius', None)
    filterSize = kwargs.get('filterSize', 0)
    normalise = kwargs.get('normalise', None)
    autoMask = kwargs.get('autoMask', True)
    mask = kwargs.get('mask', True)
    background = kwargs.get('background', None)
    
    if autoMask:
        img = pybundle.auto_mask(img, radius = radius)
    # Find the cores in the calibration image
    coreX, coreY = pybundle.find_cores(img, coreSize)
    coreX = np.round(coreX).astype('uint16')
    coreY = np.round(coreY).astype('uint16')

    # Default values
    if centreX is None:
        centreX = np.mean(coreX)
    if centreY is None:
        centreY = np.mean(coreY)
    if radius is None:
        dist = np.sqrt((coreX - centreX)**2 + (coreY - centreY)**2)
        radius = max(dist)
        
    # Delaunay triangulation and find barycentric co-ordinates for each pixel
    t1 = time.perf_counter()   
    
    calib = pybundle.init_tri_interp(img, coreX, coreY, centreX, centreY, radius, gridSize, filterSize= filterSize, background = background, normalise = normalise, mask = mask)
    #print("Init tri interp " + str(time.perf_counter() - t1))

    calib.nCores = np.shape(coreX)
    
    return calib

 
def init_tri_interp(img, coreX, coreY, centreX, centreY, radius, gridSize, **kwargs):
    """ Used by calib_tri_interp to perform Delaunay triangulation of core positions, 
    and find each pixel of reconstruction grid in barycentric co-ordinates w.r.t. 
    enclosing triangle.
    
    Returns instance of BundleCalibration.
    
    Arguments:
         img     : calibration image as 2D (mono) or 3D (colour) numpy array
         coreX   : x centre of each core as 1D numpy array
         coreY   : y centre of each core as 1D numpy array
         centreX : x centre location of bundle (reconstruction will be centred on this)
         centreY : y centre location of bundle (reconstruction will be centred on this)
         radius  : radius of bundle (reconstruction will cover a square out to this radius)
    
    Keyword Arguments:    
         gridSize   : output size of image, supply a single value, image will be square
         filterSize : optional, sigma of Gaussian filter applied, defaults to no filter
         background : optional, image used for background subtractionn as 2D numpy array
         normalise  : optional, image used for normalisation, as 2D numpy array. Can be same as 
                      calibration image, defaults to no normalisation
         mask       : optional, boolean, when reconstructing output image will be masked outside 
                      of bundle, defaults to True    
    """
    
    filterSize = kwargs.get('filterSize', None)
    normalise = kwargs.get('normalise', None)
    background = kwargs.get('background', None)
    mask = kwargs.get('mask', True)
    numba = kwargs.get('numba', True)

    if img.ndim > 2:    # Colour image if we have a third dimension to image
        col = True
    else:
        col = False

    # Delaunay triangulation over core centre locations
    points = np.vstack((coreX, coreY)).T
    #t1 = time.perf_counter()
    tri = Delaunay(points)
    #print("Delauany time ", str(time.perf_counter() - t1))

    # Make a vector of all the pixels in the reconstruction grid
    xPoints = np.linspace(centreX - radius, centreX + radius, gridSize)
    yPoints = np.linspace(centreY - radius, centreY + radius, gridSize)
    nPixels = gridSize**2
    mX, mY = np.meshgrid(xPoints, yPoints)
    interpPoints = np.vstack( ( np.reshape(mX, nPixels) , np.reshape(mY,nPixels) )).T

    # Find enclosing triangle for each pixel in reconstruction grid
    #t1 = time.perf_counter()
    mapping = tri.find_simplex(interpPoints, bruteforce=False, tol=None)
    #print("Find Simplex time ", str(time.perf_counter() - t1))

    # Write each pixel position in terms of barycentric co-ordinates w.r.t
    # enclosing triangle.
    #t1 = time.perf_counter()
    baryCoords = barycentric(nPixels, tri, mapping, interpPoints)
    #print("Barycentric time ", str(time.perf_counter() - t1))

    # Store background values
    if normalise is not None:
        normaliseVals = pybundle.core_values(normalise, coreX, coreY, filterSize).astype('double')
        if col:
            normaliseVals = np.mean(pybundle.core_values(normalise, coreX, coreY, filterSize).astype('double'),1)
            normaliseVals = np.expand_dims(normaliseVals,1)
    else:
        normaliseVals = 0

    if background is not None:
        backgroundVals = pybundle.core_values(background, coreX, coreY, filterSize).astype('double')
    else:
        backgroundVals = 0

    # The calibration is stored in in instance of BundleCalibration
    calib = BundleCalibration()
    calib.col = col
    if calib.col:
        calib.nChannels = np.shape(img)[2]

        
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
    
    Arguments:
         nPixels      : Number of pixels in grid 
         tri          : Delaunay triangulation
         mapping      : Output from Delaunay.find_simplex, records the co-ordinates of the
                        three triangle vertices surrounding each point
         interpPoints : Cartesian grid co-ordinates
    """
    # Left here as could try numba optimisation in future
    # baryCoords = np.zeros([nPixels, 3])
    #  c = np.zeros([3,2,nPixels])
    # for i in range(nPixels):
    #     c[:,:,i] = tri.transform[mapping[i]]
                                                 
    b0 = (tri.transform[mapping, :2].transpose([1, 0, 2]) *
          (interpPoints - tri.transform[mapping, 2])).sum(axis=2).T
    baryCoords = np.c_[b0, 1 - b0.sum(axis=1)]                                              
                                                  
    return baryCoords


def tri_interp_normalise(calibIn, normalise):
    """ Updates a calibration with a new normalisation without requiring
    full recalibration.
    
    Returns updated instance of BundleCalibration
    
    Arguments:
         calibIn   : input calibration, instance of BundleCalibration
         normalise : normalisation image as 2D/3D numpy array, or None to remove normalisation
    """
    calibOut = calibIn

    if normalise is not None:
        calibOut.normaliseVals = pybundle.core_values(normalise, calibOut.coreX, calibOut.coreY,calibOut.filterSize).astype('double')
        calibOut.normalise = normalise

    else:
        calibOut.normaliseVals = 0
        calibOut.normalise = None
    
    return calibOut


def tri_interp_background(calibIn, background):
     """ Updates a calibration with a new background without requiring
     full recalibration.
     
     Returns updated instance of BundleCalibration

     Arguments:
          calibIn: bundle calibration, instance of BundleCalibration
          background: background image as 2D/3D numpy array
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
     calibration using calib_tri_interp.
     
     Returns reconstructed image as 2D/3D numpy array.

     Arguments:
          img: raw image to be reconstructed as 2D (mono) or 3D (colour) numpy array
          calib: bundle calibration as instance of BundleCalibration
          
     Keyword Arguments:     
          numba: optional, if true use JIT acceleration using Numba, default is False
     """
    
     numba = kwargs.get('numba', True)

     # Extract intensity from each core
     t1 = time.perf_counter()
     cVals = pybundle.core_values(
         img, calib.coreX, calib.coreY, calib.filterSize, **kwargs).astype('float64')
     
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

     # Vector of pixels now has to be converted to a 2D/3D image
     if calib.col:
        pixelVal = np.reshape(pixelVal, (calib.gridSize, calib.gridSize, calib.nChannels))
     else:
        pixelVal = np.reshape(pixelVal, (calib.gridSize, calib.gridSize))

     if calib.mask is not None:
         if numba and numbaAvailable:
             pass   # masking is done in one step by grid_data
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
    
    Returns value of each pixel in the reconstruction grid, as 1D numpy array

    Arguments:
         baryCoords  : barycentric co-ordinates of each pixel in reconstruction grid
                       as 2D numpy array of size (3, num_pixels)
         cVals       : intensity values from each core as 1D numpy array
         coreIdx     : the indices of the three cores making up the enclosing triangle
                       for each reconstruction grid pixel, as 2D numpy array of 
                       size (3, num_pixels)
         mapping     : pixel to triangle mapping, as generated by init_tri_interp              
    """
   
    if cVals.ndim == 2:
        pixelVal = np.zeros((np.shape(baryCoords)[0], np.shape(cVals)[1]))
        for iChannel in range(np.shape(cVals)[1]):
            val = baryCoords * cVals[:, iChannel][coreIdx]
            pixelVal[:,iChannel] = np.sum(val, 1)
    else:   
        val = baryCoords * cVals[coreIdx]    
        pixelVal = np.sum(val, 1)


    pixelVal[mapping < 0] = 0
     
    return pixelVal
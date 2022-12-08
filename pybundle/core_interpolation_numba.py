# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains functions related to triangular linear interpolation 
between cores which are accelerated using the JIT functionality of the numba
package.

@author: Mike Hughes
Applied Optics Group, University of Kent
https://github.com/mikehugheskent
"""


import numba
from numba import jit, njit
import numpy as np

@jit(nopython = True)   
def core_value_extract_numba(img, coreX, coreY):
    """ Extract intensity of each core in fibre bundle image using JIT compiler
    
    :param coreX: 1D numpy array giving x co-ordinates of core centres
    :param coreY: 1D numpy array giving y co-ordinates of core centres
    :param filterSize: sigma of Gaussian filter
    :param numba: optional, if true numba JIT used for faster executio, defaults to False.
    :return: core intensity values as 1D numpy array 
    """
    
    nCores = np.shape(coreX)[0]
    cInt = np.zeros((nCores),dtype=numba.int64)
    for i in range(nCores):
        cInt[i] = (img[coreY[i], coreX[i]])
    return cInt


@jit(nopython=True)
def grid_data_numba(baryCoords, cVal, coreIdx, mapping, mask):
    """ Numba optimsied version of grid_data. 
    
    Produces pixel values based on linear interpolation between supplied
    surrounding triangle vertex core intensity values. The core values, cVals, 
    are supplied as a 1D numpy array, giving the intensity extraacted from each core.
    The indices of the three cores surrounding each pixel in the reconstruction
    grid are given by coreIdx, a 2D  numpy array. The mapping, which provides
    the tringle index for each reconstruction grid pixel, is used here purely
    for intentying pixels which lie outside the complex hull of the bundle, so
    that these can be set to 0.  
    :param baryCoords: barycentric co-ordinates of each pixel in reconstruction grid
        as 2D numpy array of size (3, num_pixels)
    :param cVals: intensity values from each core as 1D numpy array
    :param coreIdx: the indices of the three cores making up the enclosing triangle
        for each reconstruction grid pixel, as 2D numpy array of size (3, num_pixels)
    :return: value of each pixel in the reconstruction grid, as 1D numpy array
    """
    pixelVal = np.zeros((np.shape(baryCoords)[0]))
    if mask is None:

        for i in range(np.shape(baryCoords)[0]):
            pixelVal[i] = baryCoords[i,0] * cVal[coreIdx[i,0]] + baryCoords[i,1] * cVal[coreIdx[i,1]]+ baryCoords[i,2] * cVal[coreIdx[i,2]]
    else: 
        for i in range(np.shape(baryCoords)[0]):
            pixelVal[i] = int(mask[i]) * ( baryCoords[i,0] * cVal[coreIdx[i,0]] + baryCoords[i,1] * cVal[coreIdx[i,1]]+ baryCoords[i,2] * cVal[coreIdx[i,2]])
    
    pixelVal[mapping < 0] = 0
   
    return pixelVal   
  
@jit(nopython=True)
def apply_mask_numba(img, mask):
    """ Numba optimised version of apply_mask. Applies a mask to an input image. 
    :param img: input image as 2D numpy array
    :param mask: mask image, 2D numpy array with 1 for pixels to keep and 0 for
      pixels to set to 0. Must be same size as img
    :return: masked image as 2D numpy array
    """
    w,h = np.shape(img)
   
    for x in range(w):
        for y in range(h):
            img[x,y] = img[x,y] * mask[x,y]
           
    return img       
    
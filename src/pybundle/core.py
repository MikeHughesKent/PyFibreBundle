# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains core functions for locating the fibre bundle, masking
and cropping the bundle, and simple methods for removing the core structure
by spatial filtering.

@author: Mike Hughes
Applied Optics Group, University of Kent
https://github.com/mikehugheskent
"""

import numpy as np
import math
import time

import cv2 as cv

import pybundle
from pybundle.bundle_calibration import BundleCalibration


def normalise_image(img, normImg):
    """Normalise image by dividing by a reference image
    :param img: input image as 2D numpy array
    :param normImg: reference image as 2D numpy array
    :return: normalised image as 2D numpy array
    """
    img = img.astype('float64')
    normImg = normImg.astype('float64')
    normImg = np.divide(img, normImg, out=np.zeros_like(img).astype('float64'), where=normImg!=0)
    return normImg

    

def g_filter(img, filter_size):
    """ Applies 2D Gaussian filter to image. The kernel size is 8 times sigma.
    :param img: input image as 2D numpy array
    :param filter_size: sigma of Gaussian filter
    :return: filtered image as 2D numpy array
    """
    kernel_size = round(filter_size * 8)              # Kernal size needs to be larger than sigma
    kernel_size = kernel_size + 1 - kernel_size % 2   # Kernel size must be odd
    img_filt = cv.GaussianBlur(img,(kernel_size,kernel_size), filter_size)
    return img_filt


def median_filter(img, filter_size):
    """Apply median filter to an image
    :param img: input image as 2D numpy array
    :param filter_size: filter size in pixels (filter is square)
    """
    imgFilt = cv.medianBlur(img, filter_size)
    return imgFilt


def find_core_spacing(img):
    """ Estimate fibre bundle core spacing using peak in 2D Fourier transform. 
    :param img: input image showing bundle as 2D numpy array
    :return: estimated core spacing in pixels
    """
    
    # Ensure image is square
    size = np.min(np.shape(img))
    img = img[:size,:size]
    
    # Look at log-scaled 2D FFT
    fd = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))
   
    # Average radial profile     
    rad = pybundle.radial_profile(fd, (np.shape(fd)[0] / 2, np.shape(fd)[1] / 2))
    
    # Smooth the radial profile and take 1st derivative
    kernel_size = 20
    kernel = np.ones(kernel_size) / kernel_size
    radSmooth = np.convolve(rad, kernel, mode='same')
    radDiff = np.diff(radSmooth)
    
    # Find the point of fastest drop
    firstMinRange = np.round(np.shape(radDiff)[0] / 10).astype(int)
    firstMin = np.argmin(radDiff[0:firstMinRange]).astype(int)
    
    # Find point after fastest drop where gradient is positive    
    startReg = int((np.argwhere(radDiff[firstMin:] >0)[0] + firstMin))
    peakPos = np.argmax(radDiff[startReg:]) + startReg
    coreSpacing = size / peakPos 

    return coreSpacing



def edge_filter(imgSize, edgePos, skinThickness):
    """ Create a 2D edge filter with cosine smoothing
    :param imgSize: size of (sqaure) images which will be processed 
    :param edgePos: spatial frequency of cut-off
    :param skinThickness: slope of edge
    :return mask: spatial frequency domain filter as a 2D numpy array
    """

    circleRadius = imgSize / edgePos
    thickness = circleRadius * skinThickness

    innerRad = circleRadius - thickness / 2
    xM, yM = np.meshgrid(range(imgSize),range(imgSize))
    imgRad = np.sqrt( (xM - imgSize/2) **2 + (yM - imgSize/2) **2)
    mask =  np.cos(math.pi / (2 * thickness) * (imgRad - innerRad))**2
    mask[imgRad < innerRad ] = 1
    mask[imgRad > innerRad + thickness] = 0
    return mask        


def filter_image(img, filt):
    """ Apply a Fourier domain filter to an image. Filter must be same size as image.
    :param img: input image (spatial domain), 2D numpy array
    :param filt: spatial frequency domain representation of filter, 2D numpy array
    :return: filtered image, 2D numpy array
    """

    fd = np.fft.fftshift(np.fft.fft2(img))
    fd = fd * filt

    return np.abs(np.fft.ifft2(np.fft.fftshift(fd)))


def find_bundle(img, **kwargs):
    """ Locate fibre bundle by thresholding and searching for largest
    connected region. Returns tuple of (centreX, centreY, radius)
    :param img: input image of fibre bundle, 2D numpy array
    :param filterSize: sigma of Gaussian filter applied to remove core pattern, default to 4
    :return: tuple of (centreX, centreY, radius)
    """
    
    filterSize = kwargs.get('filterSize', 4)
    
    # Filter to minimise effects of structure in bundle
    imgFilt = g_filter(img, filterSize)
    imgFilt = (imgFilt / np.max(imgFilt) * 255).astype('uint8')
    
    # Threshold to binarise and then look for connected regions
    thres, imgBinary = cv.threshold(imgFilt,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #plt.imshow(imgBinary)
    num_labels, labels, stats, centroid  = cv.connectedComponentsWithStats(imgBinary, 8, cv.CV_32S)
    
    # Region 0 is background, so find largest of other regions
    sizes = stats[1:,4]
    biggestRegion = sizes.argmax() + 1
    
    # Find distance from centre to each edge and take minimum as safe value for radius
    centreX = round(centroid[biggestRegion,0]) 
    centreY = round(centroid[biggestRegion,1])
    radius1 = centroid[biggestRegion,0] - stats[biggestRegion,0]
    radius2 = centroid[biggestRegion,1] - stats[biggestRegion,1]
    radius3 = -(centroid[biggestRegion,0] - stats[biggestRegion,2]) + stats[biggestRegion,0]
    radius4 = -(centroid[biggestRegion,1] - stats[biggestRegion,3]) + stats[biggestRegion,1]
    radius = round(min(radius1, radius2, radius3, radius4))          
          
    return centreX, centreY, radius



################ MASKING AND CROPPING ###################################
    

def crop_rect(img,loc):
    """Extracts a square around the bundle using specified co-ordinates.
    :param img: input image as 2D numpy array
    :param loc: location to crop, specified as bundle location tuple of (centreX, centreY, radius)
    :return: tuple of (cropped image as 2D numpy array, new location tuple)
    """
    cx,cy, rad = loc
    imgCrop = img[cy-rad:cy+ rad, cx-rad:cx+rad]
    
    # Correct the co-ordinates of the bundle so that they
    # are correct for new cropped image
    newLoc = [rad,rad,loc[2]]
   
    return imgCrop, newLoc




def get_mask(img, loc):
    """ Returns a circular mask, 1 inside bundle, 0 outside bundle using specified
    bundle co-ordinates. Mask image has same dimensions as input image.
    :param img: img used to determine size of mask, 2D numpy array
    :param loc: location of bundle used to determine location of mask, tuple of (centreX, centreY, radius)
    :return: mask as 2D numpy array
    """
    cx, cy, rad = loc
 
    mY,mX = np.meshgrid(range(img.shape[0]),range(img.shape[1]))
   
    m = np.square(mX - cx) +  np.square(mY - cy)   
    imgMask = np.transpose(m < rad**2)
     
    return imgMask

    
def apply_mask(img, mask):
    """Sets all pixels outside bundle to 0 using a pre-defined mask. 
    :param img: input image as 2D numpy array
    :param mask: mask as 2D numy array with same dimensions as img, with areas to be kept as 1 and areas to be masked as0.
    """

    imgMasked = np.multiply(img, mask)
    return imgMasked
  

def auto_mask(img, **kwargs):
    """ Locates bundle and sets pixels outside to 0   
    :param img: input image as 2D numpy array
    :param loc: optional location of bundle as tuple of (centreX, centreY, radius), defaults to determining this using find_bundle
    :return: masked image as 2D numpy array
    """
    loc = pybundle.find_bundle(img, **kwargs)
    mask = pybundle.get_mask(img, loc)
    imgMasked = pybundle.apply_mask(img, mask)
    return imgMasked

    
def auto_mask_crop(img, **kwargs):
    """ Locates bundle, sets pixels outside to 0 and returns cropped image
    around bundle
    :param img: input image as 2D numpy array
    :param loc: optional location of bundle as tuple of (centreX, centreY, radius), defaults to determining this using find_bundle
    :return: masked and cropped image as 2D numpy array
    """
    loc = pybundle.find_bundle(img, **kwargs)
    imgMasked = pybundle.auto_mask(img)
    imgCropped = pybundle.crop_rect(imgMasked, loc)
    return imgCropped            



def crop_filter_mask(img, loc, mask, filterSize, **kwargs):
    """ For convenient quick processing of images. Sequentially crops image to 
    bundle, applies Gaussian filter and then sets pixels outside bundle to 0.
    Set loc to None to automatically locate bundle. Optional parameter resize
    allows the output images to be rescaled. If using autolocate, can
    optionally specify find_bundle options as additional arguments.    
    :param img: input image, 2D numpy array
    :param loc: location of bundle as tuple of (centreX, centreY, radius), set to None to determining this using find_bundle
    :param mask: 2D numpy array with value of 1 inside bundle and 0 outside bundle
    :param filterSize: sigma of Gaussian filter
    :param resize: optional size to rescale output image to, if not specified there will be no resize
    :return: output image as 2D numpy array
    """
    
    resize = kwargs.get('resize', False)
    if loc is None:
        loc = pybundle.find_bundle(img, **kwargs)
    img = pybundle.g_filter(img, filterSize)
    img = pybundle.apply_mask(img, mask)
    img, newLoc = pybundle.crop_rect(img, loc)
    if resize is not False:
        img = cv.resize(img, (resize,resize))        
    
    return img


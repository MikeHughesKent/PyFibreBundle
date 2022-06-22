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
import matplotlib.pyplot as plt
import time

import cv2 as cv

import pybundle
from pybundle.bundle_calibration import BundleCalibration


# Normalise image by dividing by a reference image
def normalise_image(img, normImg):
    img = img.astype('float64')
    normImg = normImg.astype('float64')
    normImg = np.divide(img, normImg, out=np.zeros_like(img).astype('float64'), where=normImg!=0)
    return normImg

    

# Applies 2D Gaussian filter to image 'img'. 'filtersize' is the sigma
# of the Gaussian. The kernel size is 8 times sigma.
def g_filter(img, filterSize):
    kernelSize = round(filterSize * 8)           # Kernal size needs to be larger than sigma
    kernelSize = kernelSize + 1 - kernelSize%2   # Kernel size must be odd
    imgFilt = cv.GaussianBlur(img,(kernelSize,kernelSize), filterSize)
    return imgFilt


# Apply median filter to an image
def median_filter(img, filterSize):
    imgFilt = cv.medianBlur(img, filterSize)
    return imgFilt


# Estimate bundle core spacing using peak in Fourier transform
def find_core_spacing(img):
    
    size = np.min(np.shape(img))
    img = img[:size,:size]
    
   
    fd = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img))))
   
            
    rad = pybundle.radial_profile(fd, (np.shape(fd)[0] / 2, np.shape(fd)[1] / 2))
    
    kernel_size = 20
    kernel = np.ones(kernel_size) / kernel_size
    radSmooth = np.convolve(rad, kernel, mode='same')
    
    #rad = np.smooth(rad,10)
    radDiff = np.diff(radSmooth)
    
    # Find the point of fastest drop
    firstMinRange = np.round(np.shape(radDiff)[0] / 10).astype(int)
    firstMin = np.argmin(radDiff[0:firstMinRange]).astype(int)
    
    # Find point after fastest drop where gradient is positive
    
    startReg = int((np.argwhere(radDiff[firstMin:] >0)[0] + firstMin))
    peakPos = np.argmax(radDiff[startReg:]) + startReg
    coreSpacing = size / peakPos 

    #plt.figure()
    #plt.plot(radSmooth)
    return coreSpacing


# Create a 2D edge filter with cosine smoothing
def edge_filter(imgSize, edgePos, skinThickness):
    circleRadius = imgSize / edgePos
    thickness = circleRadius * skinThickness

    innerRad = circleRadius - thickness / 2
    xM, yM = np.meshgrid(range(imgSize),range(imgSize))
    imgRad = np.sqrt( (xM - imgSize/2) **2 + (yM - imgSize/2) **2)
    mask =  np.cos(math.pi / (2 * thickness) * (imgRad - innerRad))**2
    mask[imgRad < innerRad ] = 1
    mask[imgRad > innerRad + thickness] = 0
    return mask        


# Apply a Fourier domain filter. Filter must be of correct size 
def filter_image(img, filt):
    fd = np.fft.fftshift(np.fft.fft2(img))
    #plt.figure()
    #plt.imshow(np.log(np.abs(fd)))
    #plt.figure()
    #plt.imshow(filt)
    fd = fd * filt

    return np.abs(np.fft.ifft2(np.fft.fftshift(fd)))

    


# Locate bundle in image 'img' by thresholding and searching for largest
# connected region. Returns tuple of (centreX, centreY, radius)
def find_bundle(img, **kwargs):
    
    filterSize = kwargs.get('filterSize', 4)
    
    # Filter to minimise effects of structure in bundle
    kernelSize = round(filterSize * 6)           # Kernal size needs to be larger than sigma
    kernelSize = kernelSize + 1 - kernelSize%2   # Kernel size must be odd
    

    imgFilt = cv.GaussianBlur(img,(kernelSize,kernelSize), filterSize)
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
    
# Extracts a square around the bundle using specified co-ordinates in tuple 
# loc = (centreX, centreY, radius)
def crop_rect(img,loc):
    cx = loc[0]
    cy = loc[1]
    rad = loc[2]
    imgCrop = img[cy-rad:cy+ rad, cx-rad:cx+rad]
    
    # Correct the co-ordinates of the bundle so that they
    # are correct for new cropped image
    newLoc = [rad,rad,loc[2]]
    #plt.imshow(imgCrop)
   
    return imgCrop, newLoc



# Returns a circular mask, 1 inside bundle, 0 outside bundle. Mask image
# has same dimensions as input image 'img'. loc = (centreX, centreY, radius)
def get_mask(img, loc):
    cx = loc[0]
    cy = loc[1]
    rad = loc[2]
    mY,mX = np.meshgrid(range(img.shape[0]),range(img.shape[1]))
   
    m = np.square(mX - cx) +  np.square(mY - cy)   
    imgMask = np.transpose(m < rad**2)
     
    return imgMask

    
# Sets all pixels outside bundle to 0 using a pre-defined mask
def apply_mask(img, mask):
    imgMasked = np.multiply(img, mask)
    return imgMasked
    
        


# Locates bundle and sets pixels outside to 0    
def auto_mask(img, **kwargs):
    
    loc = pybundle.find_bundle(img, **kwargs)
    mask = pybundle.get_mask(img, loc)
    imgMasked = pybundle.apply_mask(img, mask)
    return imgMasked

    
# Locates bundle, sets pixels outside to 0 and returns cropped image
# around bundle
def auto_mask_crop(img, **kwargs):
    loc = pybundle.find_bundle(img, **kwargs)
    imgMasked = pybundle.auto_mask(img)
    imgCropped = pybundle.crop_rect(imgMasked, loc)
    return imgCropped            


# For convenient quick processing of images. Sequentially crops image to 
# bundle, applies Gaussian filter and then sets pixels outside bundle to 0.
# Set loc to None to automatically locate bundle. Optional keyword of 
# resize = [size] to also resize the image. If using autolocate, can
# specify findBundle options in kwargs.
def crop_filter_mask(img, loc, mask, filterSize, **kwargs):
    
    resize = kwargs.get('resize', False)
    if loc is None:
        loc = pybundle.find_bundle(img, **kwargs)
    img = pybundle.g_filter(img, filterSize)
    img = pybundle.apply_mask(img, mask)
    img, newLoc = pybundle.crop_rect(img, loc)
    if resize is not False:
        img = cv.resize(img, (resize,resize))        
    
    return img


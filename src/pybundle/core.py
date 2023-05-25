# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains core functions for locating the fibre bundle, masking
and cropping the bundle, and simple methods for removing the core structure
by spatial filtering.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import numpy as np
import scipy.fft
import math
import time

import cv2 as cv

import pybundle
from pybundle.bundle_calibration import BundleCalibration
from pybundle.utility import average_channels, max_channels, extract_central


def normalise_image(img, normImg, outputType = None):
    """Normalise image by dividing pixelwise by a reference image. 
    
    Returns the normalised image as a 2D/3D numpy array. 
    
    If the image is 3D (colour) then the reference image can either by 2D 
    (in which case it will be applied to each colour plane) or 3D.
    
    Optionally specify an output type to cast to. If no output type is 
    specified, the output image will be float32. If uint8 or uint16 is 
    specified, the image will be scaled to use the full range of that data
    type.
    
    Arguments:
        img     : 2D/3D numpy array, input image
        normImg : 2D/3D numpy array, reference image to divide by
        
    Keyword Arguments:
        outputType : str, if specified, output image will be cast to this type
                     Default is None, in which case it will be returned as
                     a float32.
    """
  
        
    # Slight speed advantage over float64
    normImg = normImg.astype('float32')
    img = img.astype('float32')

    # If the normalisation image is 2D but image is 3D, we apply normalisation
    # image to each plane
    if img.ndim == 3 and normImg.ndim == 2:
        normImg = np.expand_dims(normImg, 2)
    
    # Divide, avoiding division by zero  
    out = np.divide(img, normImg, where=normImg!=0)
  
    # Below we adjust the output type if request using the optional
    # keyword. If it is an integer type we normalise to use the full
    # range
    if outputType == 'uint8':
        normImg = normImg / np.max(normImg) * 256
    
    if outputType == 'uint16':
        normImg = normImg / np.max(normImg) * 65536
        
    if outputType is not None:
        if normImg.dtype != outputType:
            normImg = normImg.astype(outputType)   
    
    return out

    

def g_filter(img, filterSize, kernelSize = None):
    """ Applies 2D Gaussian filter to image. By default
    the kernel size is 4 times the filter_size (sigma).
    
    Returns filtered image as numpy array.
    
    Arguments:        
        img          : input image as 2D/3D numpy array
        filterSize   : float, sigma of Gaussian filter
   
    Keyword Arguments:   
        kernelSize   : int, size of convolution kernal

    """
    
    # Kernal size needs to be larger than sigma
    if kernelSize is None:
        kernelSize = round(filterSize * 4)             
    
    # Force kernel size to be odd
    kernelSize = kernelSize + 1 - kernelSize % 2       
    
    img_filt = cv.GaussianBlur(img,(kernelSize, kernelSize), filterSize)
    
    return img_filt


def median_filter(img, filterSize):
    """ Applies 2D median filter to an image.
    
    Returns the filtered image as a 2D numpy array.
    
    Arguments:        
        img          : input image as 2D/3D numpy array
        filterSize   : float, sigma of Gaussian filter
   
    """
    
    imgFilt = cv.medianBlur(img, filterSize)
   
    return imgFilt


def find_core_spacing(img):
    """ Estimates fibre bundle core spacing using peak in 2D Fourier transform. 
    
    If the image is not square, a square will be cropped from the centre. 
    It is therefore usually best to crop the image to the bundle 
    before passing it to this function.
    
    Returns core spacing as float.
    
    Arguments: 
        img  : input image showing bundle as 2D/3D numpy array
        
    """
    
    # If colour image, we take the mean across the channels
    imgAv = average_channels(img)
    
    # Ensure image is square
    imgAv = extract_central(imgAv)
    
    # Look at log-scaled 2D FFT
    fd = np.log(np.abs(np.fft.fftshift(np.fft.fft2(imgAv))))
   
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
    coreSpacing = np.shape(imgAv)[0] / peakPos 
    
    return coreSpacing



def edge_filter(imgSize, edgePos, skinThickness):
    """ Creates a 2D edge filter with cosine smoothing.
    
    Returns the filter in spatial frequency domain filter as a 2D numpy array.
   
    Arguments: 
        imgSize       : size of (square) images which will be processed, and
                        size of filter output
        edgePos       : spatial frequency of cut-off
        skinThickness : slope of edge, the distance, in spatial frequency, over 
                        which it goes from 90% to 10%                        
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
    """ Applies a Fourier domain filter to an image, such as created by
    edge_filter(). Filter must be same size as image (x and y) but not
    multi-channel (i.e. a 2D array).
    
    Returns filtered image as 2D/3D numpy array.
    
    Arguments: 
        img    : input image (spatial domain), 2D/3D numpy array
        filt   : spatial frequency domain representation of filter, 2D numpy array        
    """

    fd = scipy.fft.fftshift(scipy.fft.fft2(img, axes = (0,1)),axes = (0,1))
    if img.ndim == 2:
        fd = fd * filt
    elif img.ndim == 3:
        fd = fd * np.expand_dims(filt, 2)  
        
    return np.abs(scipy.fft.ifft2(scipy.fft.fftshift(fd, axes = (0,1)), axes = (0,1)))


def find_bundle(img, **kwargs):
    """ Locate fibre bundle by thresholding and searching for largest
    connected region. 
    
    Returns tuple of (centreX, centreY, radius).
    
    Arguments: 

        img        : input image of fibre bundle, 2D numpy array
        
    Keyword Arguments:   
   
        filterSize : sigma of Gaussian filter applied to remove core pattern, 
                     defaults to 4
    """
    
    filterSize = kwargs.get('filterSize', 4)
    
    # If we have a colour image, take max value across channels
    imgFilt = max_channels(img)
    
    # Filter to minimise effects of structure in bundle
    imgFilt = g_filter(imgFilt, filterSize)
    imgFilt = (imgFilt / np.max(imgFilt) * 255).astype('uint8')
    
    # Threshold to binarise and then look for connected regions
    thres, imgBinary = cv.threshold(imgFilt,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
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
    

def crop_rect(img, loc):
    """Extracts a square around the bundle using specified co-ordinates. If the
    rectange is larger than the image then the returned image will be a rectangle,
    limited by the extent of the image.
    
    Returns tuple of (cropped image as 2D numpy array, new location tuple)

    Arguments:  
        img  : input image as 2D numpy array
        loc  : location to crop, specified as bundle location tuple of 
               (centreX, centreY, radius)
    """
    if loc is not None:
        
        h,w = np.shape(img)[:2]
        cx,cy, rad = loc
        
        minX = np.clip(cx-rad, 0, None)
        maxX = np.clip(cx+rad, None, w)
        
         
        minY = np.clip(cy-rad, 0, None)
        maxY = np.clip(cy+rad, None, h)
        
        imgCrop = img[minY:maxY, minX: maxX]
        
        # Correct the co-ordinates of the bundle so that they
        # are correct for new cropped image
        newLoc = [rad,rad,loc[2]]
       
        return imgCrop, newLoc
    else:
        return img, None



def get_mask(img, loc):
    """ Returns a circular mask, 1 inside bundle, 0 outside bundle, using specified
    bundle co-ordinates. Mask image has same dimensions as first two
    dimensions of input image (i.e. does not return a mask for each colour plane).
    
    Returns mask as 2D numpy array.
    
    Arguments:  
         img  : img used to determine size of mask, 2D numpy array
         loc  : location of bundle used to determine location of mask, 
                tuple of (centreX, centreY, radius)
    """
    cx, cy, rad = loc
 
    mY,mX = np.meshgrid(range(img.shape[0]), range(img.shape[1]))
   
    m = np.square(mX - cx) +  np.square(mY - cy)   
    imgMask = np.transpose(m < rad**2)
     
    return imgMask

    
def apply_mask(img, mask):
    """Sets all pixels outside bundle to 0 using a pre-defined mask. If the
    image is 3D, the mask will be applied to each colour plane.
    
    Arguments:  
         img   : input image as 2D numpy array
         mask  : mask as 2D numy array with same dimensions as img, 
                 with areas to be kept as 1 and areas to be masked as 0.
    """
    
    if mask is not None:
        if img.ndim == 3:
            m = np.expand_dims(mask, 2)
        else:
            m = mask
        imgMasked = np.multiply(img, m)
        
        return imgMasked
    else:
        return img
    

def auto_mask(img, loc = None, **kwargs):
    """ Locates bundle and sets pixels outside to 0 .
    
    Arguments:  
        img  : input image as 2D numpy array
    
    Keyword Arguments:
        loc    : optional location of bundle as tuple of (centreX, centreY, radius), 
                 defaults to determining this using find_bundle
        radius : optional, int, radius of mask to use rather than the automatically
                 determined radius        
        Others : if loc is not specified, other optional keyword arguments will
                 be passed to find_bundle.


    """
    radius = kwargs.get('radius', None)
    
    # If location not specified, find it
    if loc is None:
        loc = pybundle.find_bundle(img, **kwargs)
    
    # If radius was specified, replace auto determined radius
    if radius is not None:
        loc = (loc[0], loc[1], radius)
        
    # Mask image    
    mask = pybundle.get_mask(img, loc)
    imgMasked = pybundle.apply_mask(img, mask)
    
    return imgMasked

    
def auto_mask_crop(img, loc = None, **kwargs):
    """ Locates bundle, sets pixels outside to 0, and returns cropped image
    around bundle.
    
    Arguments:  

        img:  input image as 2D numpy array
        
    Keyword Arguments:
    
        loc:     optional location of bundle as tuple of (centreX, centreY, radius), 
                 defaults to determining this using find_bundle
        Others : if loc is not specified, other optional keyword arguments will
                 be passed to find_bundle.
    """
    
    if loc is None:
        loc = pybundle.find_bundle(img, **kwargs)
    imgMasked = pybundle.auto_mask(img)
    imgCropped = pybundle.crop_rect(imgMasked, loc)
    
    return imgCropped            



def crop_filter_mask(img, loc, mask, filterSize, resize = False, **kwargs):
    """ For convenient quick processing of images. Sequentially crops image to 
    bundle, applies Gaussian filter and then sets pixels outside bundle to 0.
    Set loc to None to automatically locate bundle. Optional parameter 'resize'
    allows the output images to be rescaled. If using auto-locate, can
    optionally specify find_bundle() options as additional keyword arguments. 
    
    Returns output image as 2D/3D numpy array
    
    Arguments:  

        img        : input image, 2D/3D numpy array
        loc        : location of bundle as tuple of (centreX, centreY, radius), 
                      set to None to determining this using find_bundle
        mask       : 2D numpy array with value of 1 inside bundle and 0 outside bundle
        filterSize : sigma of Gaussian filter
        
    Keyword Arguments:
    
        resize     : size to rescale output image to, default is no resize
        Others     : if loc is not specified, other optional keyword arguments will
                    be passed to find_bundle.
        
    """
    
    if loc is None:
        loc = pybundle.find_bundle(img, **kwargs)
    
    img = pybundle.g_filter(img, filterSize)
    
    img = pybundle.apply_mask(img, mask)
    
    img, newLoc = pybundle.crop_rect(img, loc)
    
    if resize is not False:
        img = cv.resize(img, (resize,resize))        
    
    return img


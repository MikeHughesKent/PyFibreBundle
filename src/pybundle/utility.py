# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.


This file contains utility functions.


@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import numpy as np
import math
import time
from PIL import Image

import cv2 as cv

def extract_central(img, boxSize = None):
    """ Extract a central square from an image. The extracted square is centred
    on the input image, with size 2 * boxSize if possible, otherwise the largest
    square that can be extracted.
    
    Returns cropped image as 2D numpy array.
    
    Arguments:
        img     : input image as 2D numpy array
        
    Keyword Arguments:    
        boxSize : size of cropping square, default is largest possible
        
    Returns:
        ndarray, cropped image
    """

    w = np.shape(img)[0]
    h = np.shape(img)[1]

    if boxSize is None:
        boxSize = min(w,h)
    cx = w/2
    cy = h/2
    boxSemiSize = min(cx,cy,boxSize)
    
    imgOut = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
    
    return imgOut


def to8bit(img, minVal = None, maxVal = None):
    """ Returns an 8 bit representation of image. If min and max are specified,
    these pixel values in the original image are mapped to 0 and 255 
    respectively, otherwise the smallest and largest values in the 
    whole image are mapped to 0 and 255, respectively.
    
    Arguments:
        img    : ndarray
                 input image as 2D numpy array
        
   Keyword Arguments:    
        minVal : float
                 optional, pixel value to scale to 0
        maxVal : float
                 optional, pixel value to scale to 255
    """
 
    img = img.astype('float64')
       
    if minVal is None:
        minVal = np.min(img)
                    
    img = img - minVal
        
    if maxVal is None:
        maxVal = np.max(img)
    else:
        maxVal = maxVal - minVal
        
    img = img / maxVal * 255
    img = img.astype('uint8')
    
    return img


def to16bit(img, minVal = None, maxVal = None):
    """ Returns an 16 bit representation of image. If min and max are specified,
    these pixel values in the original image are mapped to 0 and 2^16 
    respectively, otherwise the smallest and largest values in the 
    whole image are mapped to 0 and 2^16 - 1, respectively.
    
    Arguments:
        img    : ndarray
                 input image as 2D numpy array
        
    Keyword Arguments:    
        minVal : float
                 optional, pixel value to scale to 0
        maxVal : float
                 optional, pixel value to scale to 2^16 - 1
                 
    Returns:
        ndarray, 16 bit image             
    """   
        
    img = img.astype('float64')
       
    if minVal is None:
        minVal = np.min(img)
                    
    img = img - minVal
        
    if maxVal is None:
        maxVal = np.max(img)
    else:
        maxVal = maxVal - minVal
        
    img = img / maxVal * (2^16 - 1)
    img = img.astype('uint16')
    
    return img


def radial_profile(img, centre):
    """Produce angular averaged radial profile through image img centred on
    centre, a tuple of (x_centre, y_centre)
    
    Returns radial profile as 1D numpy array

    Arguments:
        img    : ndarray
                 input image as 2D numpy array
        centre : (int, int)
                 centre point for radial profile, tuple of (x,y)  
    Returns:
        ndarray, 1D profile             
    """
    
    y, x = np.indices((img.shape))
    r = np.sqrt((x - centre[1])**2 + (y - centre[0])**2)
    r = r.astype(int)
    
    tbin = np.bincount(r.ravel(), weights = img.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin  / nr
    
    return radialprofile 
 



def save_image8(img, filename):
    """ Saves image as 8 bit tif without scaling.
    
    Arguments:
         img      : ndarray, 
                    input image as 2D numpy array
                   
         filename : str
                    path to save to, folder must exist
    """
    
    im = Image.fromarray(img.astype('uint8'))
    im.save(filename)



def save_image16(img, filename):
    """ Saves image as 16 bit tif without scaling.
        
    Arguments:
         img      : ndarray, 
                    input image as 2D numpy array
                   
         filename : str
                    path to save to, folder must exist
    """
    im = Image.fromarray(img.astype('uint16'))
    im.save(filename)


     
def save_image8_scaled(img, filename):
    """ Saves image as 8 bit tif with scaling to use full dynamic range.
            
    Arguments:
         img      : ndarray, 
                    input image as 2D numpy array
                   
         filename : str
                    path to save to, folder must exist
    """
    
    im = Image.fromarray(to8bit(img))
    im.save(filename)
    
    
def save_image16_scaled(img, filename):
    """ Saves image as 16 bit tif with scaling to use full dynamic range.
            
    Arguments:
         img      : ndarray, 
                    input image as 2D numpy array
                   
         filename : str
                    path to save to, folder must exist
    """
        
    im = Image.fromarray(to16bit(img)[0])
    im.save(filename) 


def average_channels(img):
    """ Returns an image which is the the average pixel value across all channels of a colour image.
    It is safe to pass a 2D array which will be returned unchanged.
    
    
    Arguments:
        img:    ndarray
                 image as 2D/3D numpy array
            
    Returns:
        ndarray, averaged image        
    """     

    if img.ndim == 3:
        return np.mean(img, 2)
    else:
        return img
    
    
def max_channels(img):
    """ Returns an image which is the the maximum pixel value across all channels of a colour image.
    It is safe to pass a 2D array which will be returned unchanged.
    
    
    Arguments:
        img:   ndarray
               image as 2D/3D numpy array
               
    Returns:
        ndarray, max value image           
    """     

    if img.ndim == 3:
        return np.max(img, 2)
    else:
        return img
    
    
def resample(img, factor):
    """ Resizes an image by a factor.
    
    Arguments:
        img    : ndarray
                 image as 2D numpy array
        factor : float,
                 resize factor        
         
    Returns:
        ndarray, resmaple images
    """    
    h,w = np.shape(img)
    img = cv.resize(img, ( int(w * factor), int(h * factor)))
    
    return img
    
    
    
    
    
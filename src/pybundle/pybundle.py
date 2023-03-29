# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains the PyBundle class which provides object oriented
usage of the key functionality.

@author: Mike Hughes
Applied Optics Group, University of Kent
https://github.com/mikehugheskent
"""


import numpy as np
import math
import time

import cv2 as cv

from pybundle.core_interpolation import *    
from pybundle.bundle_calibration import BundleCalibration


class PyBundle:
       
    background = None
    normaliseImage = None
    loc = None
    mask = None
    crop = False
    filterSize = None
    coreMethod = None
    autoContrast = False
    outputType = 'uint16'
    doAutoMask = True
    calibImage = None
    coreSize = 3
    gridSize  = 512
    calibration = None
    useNumba = True  
    
    # Super Resolution
    superRes = False
    srShifts = None
    srCalibImages = None
    calibrationSR = None
    srBackgrounds = None
    srNormalisationImgs = None
    srNormToBackgrounds = False
    srNormToImages = True
    srMultiBackgrounds = False
    srMultiNormalisation = False
    srDarkFrame = None
    
    # Constants for core processing method
    FILTER = 1
    TRILIN = 2
    EDGE_FILTER = 3   
    
    
    def __init__(self):
        """ Initialise a PyBundle object, for OOP functionality of the pybundle package."""
        pass
               

    def set_filter_size(self, filterSize):
        """ Set the size of Gaussian filter used if filtering method employed"""
        self.filterSize = filterSize
        
    
    def set_bundle_loc(self, loc):
        """ Store the location of the bundle, requires tuple of (centreX, centreY, radius)."""
        self.loc = loc
        
    
    def set_core_method(self, coreMethod):
        """ Set the method to use to remove cores, FILTER, TRILIN or EDGE_FILTER"""
        self.coreMethod = coreMethod
        
        
    def set_core_size(self, coreSize):
        """ Set the estimated centre-centre core spacing used to help find cores as part of TRILIN method"""
        self.coreSize = coreSize
        
    
    def set_mask(self, mask):
        """ Provide a mask to be used. Mask must be a 2D numpy array of same size as images to be processed"""
        self.mask = mask
        
    
    def set_auto_contrast(self, ac):
        """ Determines whether images are scaled to be between 0-255. Boolean"""
        self.autoContrast = ac
        
        
    def set_crop(self, crop):
        """ Determines whether images are cropped to size of bundle (FILTER, EDGE_FILTER methods) Boolean."""
        self.crop = crop    
    
    
    def set_auto_mask(self, img, **kwargs):
        """ Automically create mask using pre-determined bundle location.
        Optionally provide a radius rather than using radius of determined
        bundle location.
        :param img: example image from which size of mask is determined
        :param radius, optional radius of mask
        """       
        if img is not None:

            if self.loc is not None:

               radius = kwargs.get('radius', self.loc[2])
               self.mask = pybundle.get_mask(img, (self.loc[0], self.loc[1], radius))
               self.doAutoMask = False
            else:
               self.mask = None
               self.doAutoMask = True   # Flag means we come back once we have an image
        else:
            self.mask = None
            self.doAutoMask = True      # Flag means we come back once we have a loc
    
        
    def create_and_set_mask(self, img, **kwargs):
        """ Determine mask from provided calibration image and set as mask
        Optionally provide a radius rather than using radius of determined
        bundle location.
        :param img: calibration image from which size of mask is determined
        :param radius, optional radius of mask
        """    
        if img is not None:
            self.set_auto_loc(img)
            
            if self.loc is not None:
                self.set_auto_mask(img, **kwargs)    
        
        
    def set_auto_loc(self, img):
        """ Calculate the location of the bundle and stores this.
        :param img: image showing bundle as 2D numpy array
        """        
        self.loc = pybundle.find_bundle(img)
                    
        
    def set_background(self, background):
        """ Store an image to be used as background. If TRILIN is being used and a 
        calibration has already been performed, the background will be added to the
        calibration.
        :param background: background image as 2D numpy array. Set as None to removed background.
        """
        if background is not None:
            self.background = background.astype('float')
        else:
            self.background = None
        if self.calibration is not None:
            self.calibration = pybundle.tri_interp_background(self.calibration, self.background)
            
        
    def set_normalise_image(self, normaliseImage):
        """ Store an image to be used for normalisation. If TRILIN is being used and a 
        calibration has already been performed, the normalisation will be added to the
        calibration.
        :param normaliseImage: normalisation image as 2D numpy array. Set as None to removed normalisation.
        """
        if normaliseImage is not None:
            self.normaliseImage = normaliseImage.astype('float')
        else:
            self.normaliseImage = None
        if self.calibration is not None:
            self.calibration = pybundle.tri_interp_normalise(self.calibration, self.normaliseImage)
            
        
    def set_output_type(self, outputType):
        """ Specify the data type of input images from 'process'. If not called, 
        default of 'uint8' will be used.
        :param outputType: one of 'uint8', 'unit16' or 'float'
        :return: True if type was valid, otherwise False
        """
        if outputType == 'uint8' or outputType == 'uint16' or outputType == 'float':
            self.outputType = outputType
            return True
        else:
            return False
        
            
    def set_calib_image(self, calibImg):
        """ Set image to be used for calibration if TRLIN method used.
        :param calibImg: calibration image as 2D numpy array
        """
        self.calibImage = calibImg.astype('float')
        
        
        
    def set_grid_size(self, gridSize):
        """ Set output image size if TRLIN method used. If not called prior to calling 'calibrate', the default value of 512 will be used.
        :param gridSize: size of square image output size
        """
        self.gridSize = gridSize
        
        
    def set_edge_filter(self, edgePos, edgeSlope):  
        """ Create filter if EDGE_FILTER method is to be used.
        :param edgePos: spatial frequency of edge in pixels of FFT of image
        :param edgeSlope: steepness of slope (range from 10% to 90%) in pixels of FFT of image
        """
        self.edgeFilter = pybundle.edge_filter(self.loc[2] *2 , edgePos, edgeSlope)
        
        
    def set_use_numba(self, useNumba):
        """ Sets whether Numba should be used for JIT compiler acceleration for functionality which support this. Boolean"""
        self.useNumba = useNumba  

    def set_sr_calib_images(self, calibImages):
        """ Provides the calibration images, a stack of shifted images used to determine shifts between images for super-resolution
        """
        self.srCalibImages = calibImages
        
        
    def set_sr_norm_to_images(self, normToImages):
        """ Sets whether super-resolution recon should normalise each input image to have the same mean intensity. Boolean"""

        self.srNormToImages = normToImages         
        
        
    def set_sr_norm_to_backgrounds(self, normToBackgrounds):
        """ Sets whether super-resolution recon should normalise each input image w.r.t. a stack of backgrounds in srBackgrounds to have the same mean intensity. Boolean"""
        self.srNormToBackgrounds = normToBackgrounds  
        
    
    def set_sr_multi_backgrounds(self, mb):
        """ Sets whether super-resolution should normalise each core in each image"""
        self.srMultiBackgrounds = mb
        
        
    def set_sr_multi_normalisation(self, mn):
        self.srMultiNormalisation = mn

        
    def set_sr_backgrounds(self, backgrounds):
        """ Provide a set of background images for background correction of each SR shifted image.
        """
        self.srBackgrounds = backgrounds  
       
        
    def set_sr_normalisation_images(self, normalisationImages):
        """ Provide a set of normalisation images for normalising intensity of each SR shifted image.
        """
        self.srNormalisationImgs = normalisationImages
        

    def set_sr_shifts(self, shifts):
        """ Provide shifts between SR images"""
        self.srShifts = shifts
     
     
    def set_sr_dark_frame(self, darkFrame):
        """ Provide a dark frame for super-resolution calibration"""
        self.srDarkFrame = darkFrame
        
    def calibrate(self):
        """ Creates calibration for TRILIN method. A calibration image, coreSize and griSize must have been set prior to calling this."""
        if self.calibImage is not None:
            self.calibration = pybundle.calib_tri_interp(self.calibImage, self.coreSize, self.gridSize, background = self.background, normalise = self.normaliseImage)
    
    
    def calibrate_sr(self):
        """ Creates calibration for TRILIN SR method. A calibration image, set of super-res shift images, coreSize and griSize must have been set prior to calling this."""
        if self.srCalibImages is not None or self.srShifts is not None:
            self.calibrationSR = pybundle.SuperRes.calib_multi_tri_interp(
                self.calibImage, self.srCalibImages,                                                                           
                self.coreSize, self.gridSize, 
                background = self.background, 
                normalise = self.normaliseImage,
                backgroundImgs = self.srBackgrounds,
                normalisationImgs = self.srNormalisationImgs,
                normToBackground = self.srNormToBackgrounds,
                normToImage = self.srNormToImages,
                shifts = self.srShifts,
                multiBackgrounds = self.srMultiBackgrounds,
                multiNormalisation = self.srMultiNormalisation,
                darkFrame = self.srDarkFrame,
                filterSize = self.filterSize)

    
    def process(self, img):
        """ Process fibre bundle image using current settings .
        :param img: input image as 2D numpy array
        :return: processing image as 2D numpy array
        """
        
        method = self.coreMethod  # to insulate against a change during processing
        
        imgOut = img        
        
        if self.doAutoMask:
            self.set_auto_mask(img)

        if method == self.FILTER:
            if self.background is not None:
                imgOut = imgOut - self.background
            if self.filterSize is not None:
                imgOut = pybundle.g_filter(imgOut, self.filterSize)
                

        if method == self.EDGE_FILTER:
            if self.background is not None:
                imgOut = imgOut - self.background
            if self.edgeFilter is not None and self.loc is not None:
                imgOut = pybundle.crop_rect(imgOut, self.loc)[0]
                imgOut = pybundle.filter_image(imgOut, self.edgeFilter)       
                
        if method == self.TRILIN and not self.superRes:
            if self.calibration is None and self.calibImage is not None:
                self.calibrate()
            if self.calibration is None: return None
            if imgOut.ndim != 2: return None
            
            imgOut = pybundle.recon_tri_interp(imgOut, self.calibration, numba = self.useNumba)
           
        if method == self.TRILIN and self.superRes:
            if self.calibrationSR is None and ( (self.srCalibImages is not None) or (self.srShifts is not None)):
                self.calibrate_sr()
 
            if self.calibrationSR is None: return None
            if imgOut.ndim != 3: return None
            if np.shape(imgOut)[2] != self.calibrationSR.nShifts: return None
            
            imgOut = pybundle.SuperRes.recon_multi_tri_interp(imgOut, self.calibrationSR, numba = self.useNumba)
       
        
        if self.autoContrast:
            t1 = time.perf_counter()
            imgOut = imgOut - np.min(imgOut)
            imgOut = imgOut / np.max(imgOut)
            if self.outputType == 'uint8':
                imgOut = imgOut * 255
            elif self.outputType == 'uint16':
                imgOut = imgOut * (2**16 - 1)
            elif self.outputType == 'float':
                imgOut = imgOut
        
        if method == self.FILTER and self.mask is not None:
            imgOut = pybundle.apply_mask(imgOut, self.mask)
            
        # Temporarily disabled because after applying the edge filter the image is
        # not the same size as the mask.a
        #if self.coreMethod == self.EDGE_FILTER and self.mask is not None:
        #    imgOut = pybundle.apply_mask(imgOut, self.mask)
            
        if method == self.TRILIN and not self.superRes and self.calibration.mask is not None:
            imgOut = pybundle.apply_mask(imgOut, self.calibration.mask)
            
        if method == self.TRILIN and self.superRes and self.calibrationSR.mask is not None:
            imgOut = pybundle.apply_mask(imgOut, self.calibrationSR.mask)
            
        if method == self.FILTER and self.crop and self.loc is not None:
            imgOut = pybundle.crop_rect(imgOut, self.loc)[0]
        
        imgOut = imgOut.astype(self.outputType)        
        
        return imgOut
    

    def get_pixel_scale(self):
        """ Returns the scaling factor between the pixel size in the raw image
        and the pixel size in the processed image. If the TRILIN method is
        selected, but a calibration has not yet been performed, returns None.
        """        
        if self.coreMethod == self.TRILIN:
            
            if not self.superRes:
                if self.calibration is not None:
                    scale = (2 * self.calibration.radius) / self.calibration.gridSize
                    return scale
                else:
                    return None
            else:
           
                if self.calibrationSR is not None:
                    scale = (2 * self.calibrationSR.radius) / self.calibrationSR.gridSize
                    return scale
                else:
                    return None
        else:    
            return 1  
        
        
    def set_super_res(self, sr):
        """ Enables or disables super resoution, sr is boolean"""
        self.superRes = sr
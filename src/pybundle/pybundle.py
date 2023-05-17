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
       
    coreMethod = None

    background = None
    normaliseImage = None
    
    autoLoc = True
    loc = None

    autoMask = True
    mask = None    
    applyMask = False
    
    crop = False
    
    filterSize = None
    
    autoContrast = False
    outputType = 'float'
    
    edgeFilter = None
    edgeFilterShape = None
    
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
    srUseLut = False
    srCalibrationLUT = None
    srParamValue = None
    #srParamCalib = None
    
    
    # Constants for core processing method
    FILTER = 1
    TRILIN = 2
    EDGE_FILTER = 3   
    
    
    def __init__(self, **kwargs):
        """ Initialise a PyBundle object, for OOP functionality of the pybundle package."""
        
        self.background = kwargs.get('backgroundImage', self.background)
        self.normaliseImage = kwargs.get('normaliseImage',self.normaliseImage)
        
        self.loc = kwargs.get('loc', self.loc )
        if self.loc is not None:
            self.autoLoc = False
        self.autoLoc = kwargs.get('autoLoc', self.autoLoc )
        
        self.applyMask = kwargs.get('applyMask', self.applyMask)

        self.mask = kwargs.get('mask', self.mask)
        if self.mask is not None:
            self.autoMask = False
        self.autoMask = kwargs.get('autoMask', self.autoMask)

        self.crop = kwargs.get('crop', self.crop)
        
        self.edgeFilterShape = kwargs.get('edgeFilterShape', self.edgeFilterShape )
        self.filterSize = kwargs.get('filterSize', self.filterSize )
        self.coreMethod = kwargs.get('coreMethod', self.coreMethod)
        self.autoContrast = kwargs.get('autoContrast', self.autoContrast)
        self.outputType = kwargs.get('outputType', self.outputType)
        self.calibImage = kwargs.get('calibImage', self.calibImage)
        self.coreSize = kwargs.get('coreSize', self.coreSize)
        self.gridSize  = kwargs.get('gridSize', self.gridSize)
        self.useNumba = kwargs.get('useNumba',self.useNumba )
        
        
        # Super Resolution
        self.superRes = kwargs.get('superRes' , self.superRes )
        self.srShifts = kwargs.get('srShifts', self.srShifts)
        self.srCalibImages = kwargs.get('srCalibImages', self.srCalibImages)
        self.srBackgrounds = kwargs.get('srBackgrounds', self.srBackgrounds)
        self.srNormalisationImgs = kwargs.get('srNormalisationImages', self.srNormalisationImgs)
        self.srNormToBackgrounds = kwargs.get('srNormToBackgrounds', self.srNormToBackgrounds)
        self.srNormToImages = kwargs.get('srNormToImages', self.srNormToImages)
        self.srMultiBackgrounds = kwargs.get('srMultiBackgrounds' , self.srMultiBackgrounds)
        self.srMultiNormalisation = kwargs.get('srMultiNormalisation' ,self.srMultiNormalisation )
        self.srDarkFrame = kwargs.get('srDarkFrame', self.srDarkFrame)
        self.srUseLut = kwargs.get('srUseLut', self.srUseLut)
        
               

    def set_filter_size(self, filterSize):
        """ Set the size of Gaussian filter used if filtering method employed"""
        self.filterSize = filterSize
        
    
    def set_loc(self, loc):
        """ Store the location of the bundle, requires tuple of (centreX, centreY, radius)."""
        self.loc = loc
        self.autoLoc = False   # We don't want to automatically find it if we have been given it
        
    
    def set_core_method(self, coreMethod):
        """ Set the method to use to remove cores, FILTER, TRILIN or EDGE_FILTER"""
        self.coreMethod = coreMethod
        
        
    def set_core_size(self, coreSize):
        """ Set the estimated centre-centre core spacing used to help find cores as part of TRILIN method"""
        self.coreSize = coreSize
        
    
    def set_mask(self, mask):
        """ Provide a mask to be used. Mask must be a 2D numpy array of same size as images to be processed"""
        self.mask = mask
        self.autoMask = False  # We don't want to automatically find it if we have been given it
        
    
    def set_auto_contrast(self, ac):
        """ Determines whether images are scaled to be between 0-255. Boolean"""
        self.autoContrast = ac
        
        
    def set_crop(self, crop):
        """ Determines whether images are cropped to size of bundle 
        for FILTER, EDGE_FILTER methods. crop is Boolean.
        """
        self.crop = crop    
    
    
    def set_apply_mask(self, applyMask):
        """ Determines whether areas outside the bundle are set to zero 
        for FILTER and EDGE_FILTER method. applyMask is Boolean.
        """
        self.apply_mask = applyMask
        
    
    def set_auto_mask(self, img, **kwargs):
        """ Set whether to automically create mask using pre-determined bundle 
        location.
        
        It is also possible to provide an image as a 2D numpy array, in which
        case the mask will be generated of the correct size for this image, but
        this is deprecates, use calibrate() instead. Optionally provide a 
        radius rather than using radius of determined bundle location.
       
        """       
        if type(img) is bool:
            if img is True:
                self.mask = None
                self.autoMask = True     
            
        # Deprecate possibility to provide an image here, avoid using for new
        # applications, use calibrate() instead.
        elif img is not None and img is not False:

            if self.loc is not None:
               radius = kwargs.get('radius', self.loc[2])
               self.mask = pybundle.get_mask(img, (self.loc[0], self.loc[1], radius))
               self.autoMask = False
          
    
        
        
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
        """ Sets whether the bundle is automatically located for cropping and masking,
        if these are turned on.
        
        It is also possible to pass an image as a 2D numpy array instead of a Boolean,
        in which case the bundle location will be determined from this image. However,
        this is not deprecated in favour of setting calibImg and then calling calibrate.
        """      
        
        if type(img) is bool:
            if img is True:
                self.loc = None
                self.autoLoc = True
        elif type(img) is np.ndarray:        
            self.loc = pybundle.find_bundle(img)
            self.autoLoc  = False
                    
        
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
        
        
    def set_edge_filter_shape(self, edgePos, edgeSlope):  
        """ Create filter if EDGE_FILTER method is to be used.
        :param edgePos: spatial frequency of edge in pixels of FFT of image
        :param edgeSlope: steepness of slope (range from 10% to 90%) in pixels of FFT of image
        """
        self.edgeFilterShape  = (edgePos, edgeSlope)
        
        
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
        
        
    def set_sr_param_value(self, val):
        """ Sets the current value of the parameter on which the shifts dependent for SR reconstruction """
        self.srParamValue = val
                
        
    def calibrate(self):
        """ Peforms calibraion steps appropriate to chosen method. A calibration 
        image must have been set prior to calling this.
        
        For TRILIN, creates inerpolation calibration.
        
        FOR FILTER, EDGE_FILTER the bundle will be located if autoLoc has been set.
        
        FOR FILTER, EDGE_FILTER the mask will be located if autoMask has been set.

        
        """
       
        assert self.calibImage is not None, "Calibration requires calibration image, use set_calib_image()."
        
        if self.coreMethod == self.TRILIN:
            if self.calibImage is not None:

                self.calibration = pybundle.calib_tri_interp(self.calibImage, self.coreSize, self.gridSize, 
                                                         background = self.background, 
                                                         normalise = self.normaliseImage,
                                                         filterSize = self.filterSize)
        
        else:
            
            if self.autoLoc and self.calibImage is not None:
                self.loc = pybundle.find_bundle(self.calibImage)
                self.autoLoc = False    
            
            if self.autoMask and self.calibImage is not None and self.loc is not None:
                self.mask = pybundle.get_mask(self.calibImage, self.loc)
                self.autoMask = False
         
        if self.coreMethod == self.EDGE_FILTER:
             assert self.loc is not None, "Calibration for edge filter requires the bundle location."
             assert type(self.edgeFilterShape) is tuple, "Edge filter shape not defined."
             self.edgeFilter = pybundle.edge_filter(self.loc[2] *2 , self.edgeFilterShape[0], self.edgeFilterShape[1])
            
    
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
        
        
        
        # If autoLoc is True (or if we are doing EDGE_FILTER), we find the location for the crop now, otherwise
        # we use the stored location (if this is None then there will be no crop)
        # We avoid doing this if we are not cropping or masking to save time
        if self.autoLoc and ((self.crop or self.coreMethod == self.EDGE_FILTER) or self.applyMask):
            if self.calibImage is not None:

                self.loc = pybundle.find_bundle(self.calibImage) 
                cropLoc = self.loc
                self.autoLoc = False   # We have done this, don't do it again
            else:
                cropLoc = pybundle.find_bundle(img) 
        else:
            cropLoc = self.loc
         
         
        # If autoMask is True, we find the location for mask crop now, otherwise
        # we use the stored mask (if this is None then there will be no mask)
        # We avoid doing this if we are not masking to save time         
        if self.autoMask and cropLoc is not None and self.applyMask is not None:
            if self.calibImage is not None:
                self.mask = pybundle.get_mask(self.calibImage, cropLoc)
                mask = self.mask
                self.autoMask = False  # We have done this, don't do it again
            else:
                mask = pybundle.get_mask(img, cropLoc)

        else:
            mask = self.mask
            
            
        # Background subtraction (This is handled separately for TRILIN)     
        if method == self.FILTER or method == self.EDGE_FILTER:
            if self.background is not None:
                imgOut = imgOut - self.background
        
        
        # Gaussian Filter
        if method == self.FILTER and self.filterSize is not None:
            imgOut = pybundle.g_filter(imgOut, self.filterSize)
            
                
       
            
                
        # Masking
        if (method == self.FILTER or method == self.EDGE_FILTER) and mask is not None:
            imgOut = pybundle.apply_mask(imgOut, mask)
       
        
        # Cropping
        if method == self.EDGE_FILTER or (method == self.FILTER and self.crop):
            if cropLoc is not None:
                imgOut = pybundle.crop_rect(imgOut, cropLoc)[0]

       
        # Edge Filter
        if method == self.EDGE_FILTER:

           if self.edgeFilter is None:
               assert cropLoc is not None, "Edge filter requires the bundle location."
               assert self.edgeFilterShape is not None, "Edge filter requires the edge filter shape to be set using set_edge_filter_shape()."
               self.edgeFilter = pybundle.edge_filter(cropLoc[2] *2 , self.edgeFilterShape[0], self.edgeFilterShape[1])
                
           if self.edgeFilter is not None and cropLoc is not None:
               imgOut = pybundle.filter_image(imgOut, self.edgeFilter) 
        
        # Normal Triangular linear interpolation    
        if method == self.TRILIN and not self.superRes:
            if self.calibration is None and self.calibImage is not None:
                self.calibrate()
            if self.calibration is None: return None
            if imgOut.ndim != 2 and imgOut.ndim != 3: return None            
            imgOut = pybundle.recon_tri_interp(imgOut, self.calibration, numba = self.useNumba)
            
         
        # Super-resolution triangular linear interpolation    
        if method == self.TRILIN and self.superRes:
            
            # Check that we have a stack of images
            if imgOut.ndim != 3: return None           
            
            # If we have a calibration LUT and have opted to use this and we have a value for the parameter, pull out the
            # correct calibration and use this for recon
          
            if self.srUseLut and self.srCalibrationLUT is not None and self.srParamValue is not None:
                calibSR = self.srCalibrationLUT.calibrationSR(self.srParamValue)
            elif self.calibrationSR is not None:
                calibSR = self.calibrationSR
            elif ( (self.srCalibImages is not None) or (self.srShifts is not None)):
                self.calibrate_sr() 
                # If we still don't have a calibration we cannot proceed    
                if self.calibrationSR is None: return None
                calibSR = self.calibrationSR
            else:
                return None
            
            # If we don't have the correct number of images in the stack, we cannot proceeed            
            if np.shape(imgOut)[2] != calibSR.nShifts: return None
            imgOut = pybundle.SuperRes.recon_multi_tri_interp(imgOut, calibSR, numba = self.useNumba)
       
           
        
        # Autocontrast
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
        
        # Type casting
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
                elif self.srCalibrationLUT is not None:
                    scale = (2 * self.srCalibrationLUT.calibrations[0].radius) / self.srCalibrationLUT.calibrations[0].gridSize
                    return scale
                else:
                    return None
        else:    
            return 1  
        
        
    def set_super_res(self, sr):
        """ Enables or disables super resoution, sr is boolean"""
        self.superRes = sr
        
        
    def set_sr_use_lut(self, useLUT):
        """ Enables or disables use of calibration LUT for super resoution, useLUT is boolean"""
        self.srUseLut = useLUT
        
    def calibrate_sr_lut(self, paramCalib, paramRange, nCalibrations) :   
        """ Creates calibration LUT for TRILIN SR method. A calibration image, set of super-res shift images, coreSize and griSize must have been set prior to calling this."""
   
        if self.srCalibImages is not None or self.srShifts is not None:
            self.srCalibrationLUT = pybundle.calibrationLUT(
                self.calibImage, self.srCalibImages,                                                                           
                self.coreSize, self.gridSize, 
                paramCalib, paramRange, nCalibrations,
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

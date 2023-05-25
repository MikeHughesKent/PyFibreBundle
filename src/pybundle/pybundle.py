# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains the PyBundle class which provides object oriented
usage of the key functionality.

@author: Mike Hughes, Applied Optics Group, University of Kent

"""


import numpy as np
import math
import time

import cv2 as cv
import pybundle

from pybundle.core_interpolation import *    
from pybundle.bundle_calibration import BundleCalibration
from pybundle.core import normalise_image, g_filter, edge_filter, filter_image, crop_rect


class PyBundle:
       
    coreMethod = None

    background = None
    normaliseImage = None
    normaliseImageFiltered = None
    normaliseImageFilterSize = None
    
    autoLoc = True
    loc = None

    autoMask = True
    mask = None    
    applyMask = False
    
    crop = False
    
    filterSize = None
    
    autoContrast = False
    outputType = 'float64'
    
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
        
        self.radius = kwargs.get('radius', None)
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
        

            

    def __process_filter(self, img, cropLoc, mask):
         """ Process fibre bundle image using FILTER method using current settings.
         Designed to be called from process.
         
         Returns processed image as 2D/3D numpy array.
         
         Arguments:
             img: input image as 2D/3D numpy array
             cropLoc : location to crop to, tuple of (centre_x, centre_y, radius)
                       (None for no crop)
             mask : mask to apply (None for no mask)
             
         """
         
         imgOut = img        
                         
         if self.background is not None: imgOut = imgOut - self.background
                 
         if self.filterSize is not None: imgOut = pybundle.g_filter(imgOut, self.filterSize)            
                   
         if self.normaliseImage is not None:
             # Check we have a filtered normaliseImage, otherwise
             # we need to do that now
             if self.normaliseImageFilterSize != self.filterSize:
                 self.normaliseImageFiltered = g_filter(self.normaliseImage, self.filterSize)
                 self.normaliseImageFilterSize = self.filterSize
             if self.filterSize is None:
                 self.normaliseImageFiltered = self.normaliseImage
             # Do the normalisation
             imgOut = normalise_image(imgOut, self.normaliseImageFiltered)         
       
         
         # Masking
         imgOut = pybundle.apply_mask(imgOut, mask)        
         
         # Cropping
         if self.crop: imgOut = pybundle.crop_rect(imgOut, cropLoc)[0]

         return imgOut
         
     
        
    def __process_edge_filter(self, img, cropLoc, mask):
        """ Process fibre bundle image using EDGE FILTER method using current settings.
        Designed to be called from process.
        
        Returns processed image as 2D/3D numpy array.
        
        Arguments:
            img: input image as 2D/3D numpy array
            cropLoc : location to crop to, tuple of (centre_x, centre_y, radius)
                      (None for no crop)
            mask : mask to apply (None for no mask)
            
        """
        
        imgOut = img
        
        if self.background is not None: imgOut = imgOut - self.background
        
         
        # Masking
        if mask is not None: imgOut = pybundle.apply_mask(imgOut, mask)
       
        
        # Cropping
        if cropLoc is not None: imgOut = pybundle.crop_rect(imgOut, cropLoc)[0]

                
        # If there isn't an edge filter created, create it now          
        if self.edgeFilter is None:
            assert cropLoc is not None, "Edge filter requires the bundle location."
            assert self.edgeFilterShape is not None, "Edge filter requires the edge filter shape to be set using set_edge_filter_shape()."
            self.edgeFilter = pybundle.edge_filter(cropLoc[2] * 2 , self.edgeFilterShape[0], self.edgeFilterShape[1])
           
        
        # Normalisation.
        # (If there is a normalisation image but not a filtered version of it
        # create a filtered version now)
        if self.normaliseImage is not None:
            
            # Create the filtered normalisation image if we don't have it
            if self.normaliseImageFilterSize != self.edgeFilterShape or self.normaliseImageFiltered is None:
                self.normaliseImageFiltered = filter_image(crop_rect(self.normaliseImage, cropLoc)[0], self.edgeFilter)
                self.normaliseImageFilterSize = self.edgeFilterShape
            
            # Do the normalisation
            imgOut = normalise_image(imgOut, self.normaliseImageFiltered)    
       
        
        # Apply edge filter
        if self.edgeFilter is not None and cropLoc is not None:
               imgOut = pybundle.filter_image(imgOut, self.edgeFilter) 
        
        return imgOut
    
    
    

    def __process_trilin(self, img):
        
        """ Process fibre bundle image using TRILIN method using current settings.
        Designed to be called from process.
        
        Returns processed image as 2D/3D numpy array.
        
        Arguments:
            img: input image as 2D/3D numpy array            
        """
        
        imgOut = img
        
        # Normal Triangular linear interpolation    
        if not self.superRes:
            if self.calibration is None and self.calibImage is not None:
                self.calibrate()
            if self.calibration is None: return None
            if imgOut.ndim != 2 and imgOut.ndim != 3: return None            
            imgOut = pybundle.recon_tri_interp(imgOut, self.calibration, numba = self.useNumba)
            
        # Super Res Triangular linear interpolation
        else:        
            
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
       
        return imgOut
    
    
    def process(self, img):
        """ Process fibre bundle image using current settings.
        
        Returns processed image as 2D/3D numpy array.
        
        Arguments:
            img: input image as 2D/3D numpy array
            
        """        
       
        # If autoLoc is still True, meaning calibrate() was not called,
        # and we need to crop, mask apply an edge filter, 
        # then we find the location for the crop now, otherwise
        # we use the stored location (if this is None then there will be no crop)
        if self.autoLoc and ((self.crop or self.coreMethod == self.EDGE_FILTER) or self.applyMask):
            if self.calibImage is not None:

                self.loc = pybundle.find_bundle(self.calibImage) 
                cropLoc = self.loc
                self.autoLoc = False   # We have done this, don't do it again
            else:
                cropLoc = pybundle.find_bundle(img) 
        else:
            cropLoc = self.loc
         
         
        # If autoMask is still True, meaning calibrate() was not called, and we 
        # are applying a mask, we create a mask now, otherwise we
        # we use the stored mask (and if this is None then there will be no mask)    
        if self.autoMask and cropLoc is not None and self.applyMask is not None:
            if self.calibImage is not None:
                self.mask = pybundle.get_mask(self.calibImage, cropLoc)
                mask = self.mask
                self.autoMask = False  # We have done this, don't do it again
            else:
                mask = pybundle.get_mask(img, cropLoc)
        else:
            mask = self.mask
            
        # Call the specific method for core removal    
        if self.coreMethod == self.FILTER: imgOut = self.__process_filter(img, cropLoc, mask)     
        if self.coreMethod == self.EDGE_FILTER: imgOut = self.__process_edge_filter(img, cropLoc, mask)     
        if self.coreMethod == self.TRILIN: imgOut = self.__process_trilin(img)
        
        # Autocontrast
        if self.autoContrast:
            imgOut = imgOut - np.min(imgOut)
            imgOut = imgOut / np.max(imgOut)
            if self.outputType == 'uint8':
                imgOut = imgOut * 255
            elif self.outputType == 'uint16':
                imgOut = imgOut * (2**16 - 1)
            elif self.outputType == 'float':
                imgOut = imgOut                
                
        # Type casting
        if imgOut.dtype != self.outputType:
            imgOut = imgOut.astype(self.outputType)        
        
        return imgOut
               

    def set_filter_size(self, filterSize):
        """ Set the size of Gaussian filter used if filtering method employed.
        
        Arguments:
            filterSize : float, sigma of Gaussian filter
        """
        self.filterSize = filterSize
        
    
    def set_loc(self, loc):
        """ Store the location of the bundle. This will also set autoLoc = False.
        
        Arguments:
            loc : bundle location, tuple of (centreX, centreY, radius)
        """
        
        self.loc = loc
        self.autoLoc = False   # We don't want to automatically find it if we have been given it
        
    
    def set_core_method(self, coreMethod):
        """ Set the method to use to remove cores, FILTER, TRILIN or EDGE_FILTER
        
        Arguments:
            coreMethod: PyBundle.FILTER, PyBundle.TRILIN or PyBundle.EDGE_FILTER
        """
        self.coreMethod = coreMethod
        
        
    def set_core_size(self, coreSize):
        """ Set the estimated centre-centre core spacing used to help find 
        cores as part of TRILIN method.
        
        Arguments:
            coreSize : float, estimate core spacing
        """
        self.coreSize = coreSize
        
    
    def set_mask(self, mask):
        """ Provide a mask to be used. Mask must be a 2D numpy array of same 
        size as images to be processed
        
        Arguments:
            mask : 2D numpy array, 1 inside bundle, 0 inside bundle. Must be
                   same size as image to be processed.
        """
        self.mask = mask
        self.autoMask = False  # We don't want to automatically find it if we have been given it
        
    
    def set_auto_contrast(self, ac):
        """ Determines whether images are scaled to be between 0-255. 
        
        Arguments:
            ac : boolean, True to autocontrast 
        """

        self.autoContrast = ac
        
        
    def set_crop(self, crop):
        """ Determines whether images are cropped to size of bundle 
        for FILTER, EDGE_FILTER methods. crop is Boolean.
        
        Arguments:
            crop : boolean, True to crop
        """
        
        self.crop = crop    
    
    
    def set_apply_mask(self, applyMask):
        """ Determines whether areas outside the bundle are set to zero 
        for FILTER and EDGE_FILTER method. 
        
        Arguments:
            applyMask: boolean, True to apply mask, False to not apply mask
        """
        self.apply_mask = applyMask
        
    
    def set_auto_mask(self, img, **kwargs):
        """ Set whether to automically create mask using pre-determined bundle 
        location.
        
        It is also possible to provide an image as a 2D numpy array, in which
        case the mask will be generated of the correct size for this image, but
        this is deprecated, use calibrate() instead. Optionally provide a 
        radius rather than using radius of determined bundle location.
        
        Arguments:
            img: boolean, True to automically create mask
            
        Keyword Arguments:
            radius : optional, int, overrides automically determined radius for
                     mask.
       
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
        
        Arguments:
            img: calibration image from which size of mask is determined
            
        Keyword Arguments:            
            radius : radius of mask, default is to determine this automtically
        """    
        if img is not None:
            self.set_auto_loc(img)
            
            if self.loc is not None:
                self.set_auto_mask(img, **kwargs)    
        
        
    def set_auto_loc(self, img):
        """ Sets whether the bundle is automatically located for cropping and masking,
        if these are turned on, depending on boolean value passed.
        
        It is also possible to pass an image as a 2D numpy array instead of a Boolean,
        in which case the bundle location will be determined from this image. However,
        this is noq deprecated in favour of setting calibImg and then calling calibrate.
        
        Arguments:
            img : boolean, True to auto-locate bundle, False to not.
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

        Arguments:

            backgroundImage : background image as 2D/3D numpy array. Set as 
                              None to remove background.        
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
        
        Arguments:

            normaliseImage : normalisation image as 2D/3D numpy array. Set as 
                             None to remove normalisation.
        """
        if normaliseImage is not None:
            self.normaliseImage = normaliseImage.astype('float')
        else:
            self.normaliseImage = None
        if self.calibration is not None:
            self.calibration = pybundle.tri_interp_normalise(self.calibration, self.normaliseImage)
            
        
    def set_output_type(self, outputType):
        """ Specify the data type of input images returned from 'process'. 
        Returns False if type not valid.
        
        Arguments:

            outputType : str, one of 'uint8', 'unit16' or 'float'
        """
        if outputType == 'uint8' or outputType == 'uint16' or outputType == 'float' or outputType == 'float32' or outputType == 'float64':
            self.outputType = outputType
            return True
        else:
            return False
        
            
    def set_calib_image(self, calibImg):
        """ Set image to be used for calibration.
        
        Arguments:
            calibImg : calibration image as 2D/3D numpy array
        """
        self.calibImage = calibImg.astype('float')
        
        
    def set_radius(self, radius):
        """ Sets the radius of the bundle in the image. This will override
        any automically determined value.
        
        Arguments:
            radius : int, bundle radius
        """
        self.radius = radius
        
        
    def set_grid_size(self, gridSize):
        """ Sets output image size if TRLIN method used. If not called prior 
        to calling 'calibrate', the default value of 512 will be used.
        
        Arguments:
            gridSize : int, size of square image output 
        """
        self.gridSize = gridSize
        
        
    def set_edge_filter_shape(self, edgePos, edgeSlope):  
        """ Creates and stores filter for EDGE_FILTER method.
        
        Arguments:
            edgePos   : float, spatial frequency of edge in pixels of FFT of image
            edgeSlope : float, steepness of slope (range from 10% to 90%) in pixels of FFT of image
        """
        
        self.edgeFilterShape  = (edgePos, edgeSlope)
        
        
    def set_use_numba(self, useNumba):
        """ Sets whether Numba should be used for JIT compiler acceleration for 
        functionality which supports this.

        Arguments:
            useNumba : boolean, True to use Numba, False to not.
        """
        
        self.useNumba = useNumba  

    def set_sr_calib_images(self, calibImages):
        """ Provides the calibration images, a stack of shifted images used to 
        determine shifts between images for super-resolution
        
        Arguments:
            calibImages : 3D numpy array, stack of shifted images.
            
        """
        self.srCalibImages = calibImages
        
        
    def set_sr_norm_to_images(self, normToImages):
        """ Sets whether super-resolution recon should normalise each input 
        image to have the same mean intensity. 
        
        Arguments:
            normToImages : boolean, True to normalise, False to not
        
        """

        self.srNormToImages = normToImages         
        
        
    def set_sr_norm_to_backgrounds(self, normToBackgrounds):
        """ Sets whether super-resolution recon should normalise each input 
        image w.r.t. a stack of backgrounds in srBackgrounds to have the same 
        mean intensity. 
        
        Arguments:
            normToBackgrounds : boolean, True to normalise, False to not
        
        """
        self.srNormToBackgrounds = normToBackgrounds  
        
    
    def set_sr_multi_backgrounds(self, mb):
        """ Sets whether super-resolution should use individual backgrounds for each
        each shifted image.
        
        Arguments:
            mb : boolean, True to use multiple backgrounds, False to not
            
        """
        self.srMultiBackgrounds = mb
        
        
    def set_sr_multi_normalisation(self, mn):
        """ Sets whether super-resolution should normalise to each core in each image
        
        Arguments:
            mn : boolean, True to normalise each core in each image, False to not
            
        """
        self.srMultiNormalisation = mn

        
    def set_sr_backgrounds(self, backgrounds):
        """ Provide a set of background images for background correction of 
        each SR shifted image.
        
        Arguments:
            backgrounds : 3D numpy array, stack of background images
            
        """
        self.srBackgrounds = backgrounds  
       
        
    def set_sr_normalisation_images(self, normalisationImages):
        """ Provide a set of normalisation images for normalising intensity of 
        each SR shifted image.
        
        Arguments:
            normalisationImages : 3D numpy array, stack of images
            
        """
        self.srNormalisationImgs = normalisationImages
        

    def set_sr_shifts(self, shifts):
        """ Provide shifts between SR images. If this is set then no registration
        will be performed. 
        
        Arguments:
            shifts : 2D numpy array, shifts for each image relative to first
                     image. 1st axis is image number, 2nd axis is (x,y)
        """
        self.srShifts = shifts
     
     
    def set_sr_dark_frame(self, darkFrame):
        """ Provide a dark frame for super-resolution calibration.
        
        Arguments:
            darkFrame : 2D numpy array, dark frame
        
        """
        self.srDarkFrame = darkFrame
        
        
    def set_sr_param_value(self, val):
        """ Sets the current value of the parameter on which the shifts
        dependent for SR reconstruction.
        
        Arguments:
            val : parameter value
        
        """
        self.srParamValue = val
                
        
    def calibrate(self):
        """ Peforms calibraion steps appropriate to chosen method. A calibration 
        image must have been set prior to calling this.
        
        For TRILIN, creates interpolation calibration.
        
        For FILTER, EDGE_FILTER, a filtered normalisation image is created.
        
        For FILTER, EDGE_FILTER the bundle will be located if autoLoc has been set.
        
        For FILTER, EDGE_FILTER the mask will be located if autoMask has been set.
        
        """
        assert self.calibImage is not None, "Calibration requires calibration image, use set_calib_image()."
        
        if self.coreMethod == self.TRILIN:
            if self.calibImage is not None:

                self.calibration = pybundle.calib_tri_interp(self.calibImage, self.coreSize, self.gridSize, 
                                                         background = self.background, 
                                                         normalise = self.normaliseImage,
                                                         filterSize = self.filterSize,
                                                         mask = True,
                                                         autoMask = True,
                                                         radius = self.radius)
        else:
            
            if self.autoLoc and self.calibImage is not None:
                self.loc = pybundle.find_bundle(self.calibImage)
                # If user has specified a radius, over-ride the auto determined
                # one from find_bundle
                if self.radius is not None:
                    self.loc = (self.loc[0], self.loc[1], self.radius)
                self.autoLoc = False    
            
            if self.autoMask and self.calibImage is not None and self.loc is not None:
                self.mask = pybundle.get_mask(self.calibImage, self.loc)
                self.autoMask = False
                
        # If we are Gaussian filtering and we have a normalisation image
        # create a filtered version of it for use later.
        if self.coreMethod == self.FILTER:
             if self.filterSize is not None and self.normaliseImage is not None:  
                 self.normaliseImageFiltered = g_filter(self.normaliseImage, self.filterSize)
                 self.normaliseImageFilterSize = self.filterSize     # So we can realise if the filtersize changes we
                                                                     # need to redo this
         
        if self.coreMethod == self.EDGE_FILTER:
             assert self.loc is not None, "Calibration for edge filter requires the bundle location."
             assert type(self.edgeFilterShape) is tuple, "Edge filter shape not defined."
             self.edgeFilter = pybundle.edge_filter(self.loc[2] *2 , self.edgeFilterShape[0], self.edgeFilterShape[1])
             # If we have a normalisation image, create a filtered version of it for use later.
             if self.normaliseImage is not None and self.loc is not None:
                 self.nomaliseImageFiltered = filter_image(crop_rect(self.normaliseImage, self.loc)[0], self.edgeFilter)
                 self.normaliseImageFilterSize = self.edgeFilterShape
                 
    
    def calibrate_sr(self):
        """ Creates calibration for TRILIN SR method. A calibration image, 
        set of super-res shift images, coreSize and gridSize must have been 
        set prior to calling this.
        """
        
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
        """ Enables or disables super resolution.
        
        Arguments:
            sr : boolean, True to use super-resolution.
        """
        self.superRes = sr
        
        
    def set_sr_use_lut(self, useLUT):
        """ Enables or disables use of calibration LUT for super resoution.
        
        Arguments:
            useLUT : boolean, True to use calibration LUT.
            
        """
        self.srUseLut = useLUT
        
    def calibrate_sr_lut(self, paramCalib, paramRange, nCalibrations) :   
        """ Creates calibration LUT for TRILIN SR method. A calibration image, 
        set of super-res shift images, coreSize and griSize must have been 
        set prior to calling this.
        
        Arguments:
            paramCalib   :  parameter shift calibration, as generated by calib_param_shift()
            paramRange   :  tuple of (min, max) defining range of parameter values to generate for
            nCalibration : int, number of parameter values to generate for
            
        """
   
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

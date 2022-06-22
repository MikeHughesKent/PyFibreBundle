# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This files contains the pybundle class which provides object oriented
usage of the key functionality of pybundle

@author: Mike Hughes
Applied Optics Group, University of Kent
https://github.com/mikehugheskent
"""


import numpy as np
import math
import matplotlib.pyplot as plt
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
    outputType = 'uint8'
    calibImage = None
    coreSize = 3
    gridSize  = 512
    calibration = None
    FILTER = 1
    TRILIN = 2
    EDGE_FILTER = 3
    
    
    
    def __init__(self):
        pass
       
        
    ######### OOP Methods ####################
    def set_filter_size(self, filterSize):
        self.filterSize = filterSize
    
    def set_bundle_loc(self, loc):
        self.loc = loc
    
    def set_core_method(self, coreMethod):
        self.coreMethod = coreMethod
            
    def set_mask(self, mask):
        self.mask = mask
    
    def set_auto_contrast(self, ac):
        self.autoContrast = ac
        
    def set_crop(self, crop):
        self.crop = crop
    
    # Automically create mask using pre-determined bundle location.
    # Optionally provide a radius rather than using radius of determined
    # bundle location
    def set_auto_mask(self, img, **kwargs):
        
        if self.loc is not None:
            radius = kwargs.get('radius', self.loc[2])
            self.mask = pybundle.get_mask(img, (self.loc[0], self.loc[1], radius))
           
        else:
            self.mask = None
    
    def set_auto_loc(self, img):
        self.loc = pybundle.find_bundle(img)
                    
    def set_background(self, background):
        if background is not None:
            self.background = background.astype('float')
        else:
            self.background = None
        if self.calibration is not None:
            self.calibration = pybundle.tri_interp_background(self.calibration, self.background)
        
    def set_normalise_image(self, normaliseImage):
        if normaliseImage is not None:
            self.normaliseImage = normaliseImage.astype('float')
        else:
            self.normaliseImage = None
        if self.calibration is not None:
            self.calibration = pybundle.tri_interp_normalise(self.calibration, self.normaliseImage)
        
    def set_output_type(self, outputType):
        if outputType == 'uint8' or outputType == 'uint16' or outputType == 'float':
            self.outputType = outputType
            
    def set_calib_image(self, calibImg):
        self.calibImage = calibImg.astype('float')
        
    def set_grid_size(self, gridSize):
        self.gridSize = gridSize
        
    def set_edge_filter(self, edgePos, edgeSlope):  
        self.edgeFilter = pybundle.edge_filter(self.loc[2] *2 , edgePos, edgeSlope)
     
    def calibrate(self):
        
        if self.calibImage is not None:
            self.calibration = pybundle.calib_tri_interp(self.calibImage, self.coreSize, self.gridSize, background = self.background, normalise = self.normaliseImage)
    
    # Process fibre bundle image using current settings    
    def process(self, img):
        imgOut = img.astype('float')
       
        if self.coreMethod == self.FILTER:
            if self.background is not None:
                imgOut = imgOut - self.background
            if self.filterSize is not None:
                imgOut = pybundle.g_filter(imgOut, self.filterSize)
        if self.coreMethod == self.EDGE_FILTER:
            if self.background is not None:
                imgOut = imgOut - self.background
            if self.edgeFilter is not None and self.loc is not None:
                imgOut = pybundle.crop_rect(imgOut, self.loc)[0]
                imgOut = pybundle.filter_image(imgOut, self.edgeFilter)
                
        if self.coreMethod == self.TRILIN:
            if self.calibration is not None:
                imgOut = pybundle.recon_tri_interp(imgOut, self.calibration)
            else:
                return None
        
        if self.autoContrast:
            imgOut = imgOut - np.min(imgOut)
            imgOut = imgOut / np.max(imgOut)
            if self.outputType == 'uint8':
                imgOut = imgOut * 255
            elif self.outputType == 'uint16':
                imgOut = imgOut * (2**16 - 1)
            elif self.outputType == 'float':
                imgOut = imgOut
        
        if self.coreMethod == self.FILTER and self.mask is not None:
            imgOut = pybundle.apply_mask(imgOut, self.mask)
            
        #if self.coreMethod == self.EDGE_FILTER and self.mask is not None:
        #    imgOut = pybundle.apply_mask(imgOut, self.mask)
            
        if self.coreMethod == self.TRILIN and self.calibration.mask is not None:
            imgOut = pybundle.apply_mask(imgOut, self.calibration.mask)
            
        if self.coreMethod == self.FILTER and self.crop and self.loc is not None:
            imgOut = pybundle.crop_rect(imgOut, self.loc)[0]
        
        
        
        imgOut = imgOut.astype(self.outputType)        
                
            
        return imgOut



   
    
    
    
    
# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains the SuperRes class which has functions for improvion
resolution of fibre bundle images by combining multiple images of the 
object shifted with respect to the bundle core pattern.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import pybundle

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv

from skimage.registration import phase_cross_correlation


class SuperRes:

    
    # Calibration step for super-resolution.
    def calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, **kwargs):

        shifts = kwargs.get('shifts', None)
        centreX = kwargs.get('centreX', None)
        centreY = kwargs.get('centreY', None)
        radius = kwargs.get('radius', None)
        filterSize = kwargs.get('filterSize', 0)
        normToImage = kwargs.get('normToImage', None)
        normToBackground = kwargs.get('normToBackground', None)
        backgroundImgs = kwargs.get('backgroundImgs', None)
        postFilterSize = kwargs.get('postFilterSize', None)
        imageScaleFactor = kwargs.get('imageScaleFactor', None)

        singleCalib = pybundle.calib_tri_interp(
            calibImg, coreSize, gridSize, **kwargs)

        # Default values
        if centreX is None:
            centreX = np.mean(singleCalib.coreX)
        if centreY is None:
            centreY = np.mean(singleCalib.coreY)
        if radius is None:
            dist = np.sqrt((singleCalib.coreX - centreX)**2 +
                           (singleCalib.coreY - centreY)**2)
            radius = max(dist)

        nImages = np.shape(imgs)[2]

        if normToImage is not None:

            imageScaleFactor = np.array(())

            refMean = np.mean(pybundle.core_values(
                imgs[:, :, 0], singleCalib.coreX, singleCalib.coreY, filterSize))

            for idx in range(nImages):
                meanVal = np.mean(pybundle.core_values(
                    imgs[:, :, idx], singleCalib.coreX, singleCalib.coreY, filterSize))
                imageScaleFactor = np.append(
                    imageScaleFactor, refMean / meanVal)

        if normToBackground is not None:
            imageScaleFactor = np.array(())

            refMean = np.mean(pybundle.core_values(
                backgroundImgs[:, :, 0], singleCalib.coreX, singleCalib.coreY, filterSize))

            for idx in range(nImages):
                meanVal = np.mean(pybundle.core_values(
                    backgroundImgs[:, :, idx], singleCalib.coreX, singleCalib.coreY, filterSize))
                imageScaleFactor = np.append(
                    imageScaleFactor, refMean / meanVal)

        imgsProc = np.zeros((gridSize, gridSize, nImages))

        if shifts is None:

            for idx in range(nImages):

                imgsProc[:, :, idx] = pybundle.recon_tri_interp(
                    imgs[:, :, idx], singleCalib)

            shifts = SuperRes.get_shifts(imgsProc, **kwargs)

        # Since we have done the shift estimation on a different sized grid
        # to the original image, need to scale the values
        shifts = shifts * radius * 2 / gridSize

        coreXList = singleCalib.coreX
        coreYList = singleCalib.coreY

        for idx, shift in enumerate(shifts[1:]):

            coreXList = np.append(
                coreXList, singleCalib.coreX + shifts[idx + 1][0])
            coreYList = np.append(
                coreYList, singleCalib.coreY + shifts[idx + 1][1])

        calib = pybundle.init_tri_interp(calibImg, coreXList, coreYList, centreX, centreY,
                                         radius, gridSize, filterSize=filterSize, background=None, normalise=None)

        # Have to set background and normalise to None in previous line as the core
        # positions are not correcround value extractions.
        # We later copy across the background/normalise values from the single image
        # calibration

        calib.coreXInitial = singleCalib.coreX
        calib.coreYInitial = singleCalib.coreY

        calib.background = singleCalib.background
        calib.normalise = singleCalib.normalise

        calib.backgroundVals = singleCalib.backgroundVals
        calib.normaliseVals = singleCalib.normaliseVals

        calib.shifts = shifts
        calib.imageScaleFactor = imageScaleFactor
        
        calib.postFilterSize = postFilterSize
        

        return calib
    
    
    
    # Reconstruction step for super-resolution.
    def recon_multi_tri_interp(imgs, calib):

        nImages = np.shape(imgs)[2]

        cVals = []

        for i in range(nImages):

            # Extract intensity from each core
            cValsThis = (pybundle.core_values(
                imgs[:, :, i], calib.coreXInitial, calib.coreYInitial, calib.filterSize).astype('double'))

            if calib.imageScaleFactor is not None:
                cValsThis = cValsThis * calib.imageScaleFactor[i]

            if calib.background is not None:

                cValsThis = cValsThis - calib.backgroundVals

            if calib.normalise is not None:
                cValsThis = cValsThis / calib.normaliseVals

            cVals = np.append(cVals, cValsThis)

        # Triangular linear interpolation
        pixelVal = np.zeros_like(calib.mapping, dtype='double')
       
     
        val =  calib.baryCoords * cVals[calib.coreIdx]
        pixelVal = np.sum(val,1)
        pixelVal[calib.mapping < 0] = 0

        # Vector of pixels now has to be converted to a 2D image
        imgOut = np.reshape(pixelVal, (calib.gridSize, calib.gridSize))
        
        if calib.postFilterSize is not None:
            imgOut = pybundle.g_filter(imgOut, calib.postFilterSize)

        return imgOut





    # Determines the shift of each image w.r.t. first images
    def get_shifts(imgs, **kwargs):

        templateSize = kwargs.get('templateSize', None)
        refSize = kwargs.get('refSize', None)
        upsample = kwargs.get('upSample', 2)

        imgSize = np.min(np.shape(imgs[:, :, 0]))

        if templateSize is None:
            templateSize = imgSize / 4

        if refSize is None:
            refSize = imgSize / 2

        nImages = np.shape(imgs)[2]
        refImg = imgs[:, :, 0]
        shifts = np.zeros((nImages, 2))
        for iImage in range(1, nImages):
            img = imgs[:, :, iImage]
            thisShift = SuperRes.find_shift(refImg, img, templateSize, refSize,upsample)

            shifts[iImage,0] = thisShift[0]
            shifts[iImage,1] = thisShift[1]
           

        return shifts
    
    
    

    # Determines shift between two images by NCC
    def find_shift(img1, img2, templateSize, refSize, upsample):

        if refSize < templateSize or min(np.shape(img1)) < refSize or min(np.shape(img2)) < refSize:
            return -1
        else:

            template = pybundle.extract_central(
                img2, templateSize).astype('float32')
            refIm = pybundle.extract_central(img1, refSize).astype('float32')
            
            if upsample != 1:

                template = cv.resize(template, (np.shape(template)[0] * upsample,np.shape(template)[1] * upsample))
                refIm = cv.resize(refIm, (np.shape(refIm)[0] * upsample, np.shape(refIm)[0] * upsample))

            res = cv.matchTemplate(template, refIm, cv.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            shift = [(max_loc[0] - (refSize - templateSize) * upsample)/upsample,
                     (max_loc[1] - (refSize - templateSize) * upsample)/upsample]
            return shift

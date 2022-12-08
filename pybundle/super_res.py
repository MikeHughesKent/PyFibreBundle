# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains the SuperRes class which has functions for improving
resolution of fibre bundle images by combining multiple images of the 
object shifted with respect to the bundle core pattern.

@author: Mike Hughes
Applied Optics Group
University of Kent
"""

import pybundle
import time

import numpy as np
import matplotlib.pyplot as plt

import cv2 as cv


# We try to import numba here and if successful, load the numba-optimised
# interpolation funactions. If we get an error (i.e. library not available)
# then we won't call the function that require this.
try:
    from numba import jit
    import numba
    from pybundle.core_interpolation_numba import *
    numbaAvailable = True
except:
    numbaAvailable = False


class SuperRes:

    def calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, **kwargs):
        """ Calibration step for super-resolution reconstruction. Either specify the
        known shifts between images, or provide an example set of images
        which the shifts will be calculated from. 

        :param calibImg: calibration image of fibre bundle, 2D numpy array
        :param imgs: example set of images with the same set of mutual shifts as the images to
            later be used to recover an enhanced resolution image from. 3D numpy array.
            Can be None if 'shifts' is specified instead.
        :param coreSize: estimate of average spacing between cores
        :param gridSize: output size of image, supply a single value, image will be square
        :param normalise: optional, image used for normalisation, as 2D numpy array. Can be same as calibration image, defaults to no normalisation
        :param background: image used for background subtraction as 2D numpy array, defaults to no background
        :param shifts: optional, known x and y shifts between images as 2D numpy array of size (numImages,2). 
                       Will override 'imgs' if specified as anything other than None.
        :param centreX: optional, x centre location of bundle, if not specified will be determined automatically
        :param centreY: optional, y centre location of bundle, if not specified will be determined automatically
        :param radius: optional, radius of bundle, if not specified will be determined automatically
        :param filterSize: optional, sigma of Gaussian filter applied when finding cores, defaults to no filter
        :param normToImage: optional, if True each image will be normalised to have the same mean intensity. Defaults to False.
        :param normToBackground: optional, if true, each image will be normalised with respect to the corresponding 
                       background image from a stack of background images (one for each shift position) provided in backgroundImgs. 
                       Defaults to False.
        :param backgroundImgs: optional, stack of images, same size as imgs which are used to normalise. Defaults to None.
        :param imageScaleFactor: If normToBackground and normToImage are False (default), use this to specify the normalisation factors for each image. Provide a 1D array the same size as the number of shifted images. Each image will be multiplied by the corresponding factor prior to reconstruction. Default is None (i.e. no scaling).
        :param autoMask: optional, mask pixels outside bundle when searching for cores. Defualts to True.
        :param mask: optional, boolean, when reconstructing output image will be masked outside of bundle. Defaults to True
        :return: instance of BundleCalibration

        """
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
        mask = kwargs.get('mask', True)

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

        if shifts is not None:
            nImages = np.shape(shifts)[0]
        else:
            nImages = np.shape(imgs)[2]

        if normToImage is not None and imgs is not None:

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


        # Have to set 'background' and 'normalise' to None in previous line as the cores are the shifted
        # positions in the full set from all the images and not the actual core position in each images.
        # We later copy across the background/normalise values from the single image calibration.

        calib.coreXInitial = singleCalib.coreX
        calib.coreYInitial = singleCalib.coreY

        calib.background = singleCalib.background
        calib.normalise = singleCalib.normalise

        calib.backgroundVals = singleCalib.backgroundVals
        calib.normaliseVals = singleCalib.normaliseVals

        calib.shifts = shifts
        calib.imageScaleFactor = imageScaleFactor

        calib.postFilterSize = postFilterSize

        if mask:
            calib.mask = pybundle.get_mask(
                np.zeros((gridSize, gridSize)), (gridSize/2, gridSize/2, gridSize/2))
        else:
            calib.mask = None

        return calib
    

    def recon_multi_tri_interp(imgs, calib, **kwargs):
        """ Reconstruct image with super-resolution from set of shifted image. Requires calibration
        to have been performed and stored in 'calib' as instance of BundleCalibration.
        :param imgs: set of shifted images
        :param calib: calibration, instance of BundleCalibration (must be
            created by calib_multi_tri_interp and not calib_tri_interp).
        :return: reconstructed image as 2D numpy array
        """
        
        numba = kwargs.get('numba', True)

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
        #pixelVal = np.zeros_like(calib.mapping, dtype='double')
        #val = calib.baryCoords * cVals[calib.coreIdx]
        #pixelVal = np.sum(val, 1)
        #pixelVal[calib.mapping < 0] = 0

        # Triangular linear interpolation
        if numba and numbaAvailable:
            if calib.mask is not None:
                maskNumba = np.squeeze(np.reshape(calib.mask, (np.product(np.shape(calib.mask)),1)))
            else:
                maskNumba = None
            pixelVal = pybundle.grid_data_numba(calib.baryCoords, cVals, calib.coreIdx, calib.mapping, maskNumba)
        else:
            pixelVal = pybundle.grid_data(calib.baryCoords, cVals, calib.coreIdx, calib.mapping)


        # Vector of pixels now has to be converted to a 2D image
        imgOut = np.reshape(pixelVal, (calib.gridSize, calib.gridSize))

        if calib.postFilterSize is not None:
            imgOut = pybundle.g_filter(imgOut, calib.postFilterSize)

        if calib.mask is not None:
            imgOut = pybundle.apply_mask(imgOut, calib.mask)

        return imgOut


    def get_shifts(imgs, **kwargs):
        """ Determines the shift of each image in a stack w.r.t. first image
        :param imgs: stack of images as 3D numpy array
        :param templateSize: a square of this size is extracted from imgs as the template
        :param refSize: a sqaure of this size is extracted from first image as the reference image. 
           Must be bigger than templateSize and the maximum shift detectable is 
           (refSize - templateSize)/2        
        :param upSample: upsampleing factor for images before shift detection for sub-pixel accuracy
        :return: shifts as 2D numpy array
        """
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
            thisShift = SuperRes.find_shift(
                refImg, img, templateSize, refSize, upsample)

            shifts[iImage, 0] = thisShift[0]
            shifts[iImage, 1] = thisShift[1]

        return shifts
    

    def find_shift(img1, img2, templateSize, refSize, upsample):
        """ Determines shift between two images by Normalised Cross Correlation (NCC). A sqaure template extracted
        from the centre of img2 is compared with a sqaure region extracted from the reference image img1. The size 
        of the template (templateSize) must be less than the size of the reference (refSize). The maximum
        detectable shift is the (refSize - templateSize) / 2.
        : param img1 : image as 2D numpy array
        : param img2 : image as 2D numpy array
        : param templateSize : size of square region of img2 to use as template. 
        : param refSize : size of square region of img1 to template match with
        : upsample : factor to scale images by prior to template matching to
                     allow for sub-pixel registration.        
        """

        if refSize < templateSize or min(np.shape(img1)) < refSize or min(np.shape(img2)) < refSize:
            return -1
        else:

            template = pybundle.extract_central(
                img2, templateSize).astype('float32')
            refIm = pybundle.extract_central(img1, refSize).astype('float32')

            if upsample != 1:

                template = cv.resize(template, (np.shape(template)[
                                     0] * upsample, np.shape(template)[1] * upsample))
                refIm = cv.resize(
                    refIm, (np.shape(refIm)[0] * upsample, np.shape(refIm)[0] * upsample))

            res = cv.matchTemplate(template, refIm, cv.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            shift = [(max_loc[0] - (refSize - templateSize) * upsample)/upsample,
                     (max_loc[1] - (refSize - templateSize) * upsample)/upsample]
            return shift

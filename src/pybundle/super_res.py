# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This file contains the SuperRes class which has functions for improving
resolution of fibre bundle images by combining multiple images of the 
object shifted with respect to the bundle core pattern.

The preferred way to use the functionality is via the PyBundle class rather 
than calling these functions directly.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import pybundle
import time

import numpy as np

import cv2 as cv


# We try to import numba here and if successful, load the numba-optimised
# interpolation funactions. If we get an error (i.e. library not available)
# then we won't call the functions that require this.
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
        
        Returns instance of BundleCalibration.
        
        Arguments:

               calibImg   : calibration image of fibre bundle, 2D numpy array
               imgs       : example set of images with the same set of mutual shifts as the images to
                            later be used to recover an enhanced resolution image from. 3D numpy array.
                            Can be None if 'shifts' is specified instead.
               coreSize   : float, estimate of average spacing between cores
               gridSize   : int, output size of image, supply a single value, image will be square
               
        Keyword Arguments:       
              
               normalise  : image used for normalisation, as 2D numpy array. Can be same as calibration image, defaults to no normalisation
               background : image used for background subtraction, as 2D numpy array, defaults to no background
               shifts     : known x and y shifts between images as 2D numpy array of size (numImages,2). 
                            Will override doing registration of 'imgs' if specified as anything other than None.
               centreX    : int, x centre location of bundle, if not specified will be determined automatically
               centreY    : int, y centre location of bundle, if not specified will be determined automatically
               radius     : int, radius of bundle, if not specified will be determined automatically
               filterSize : float, sigma of Gaussian filter applied when finding cores, defaults to no filter
               normToImage: boolean, if True each image will be normalised to have the same mean intensity. Defaults to False.
               normToBackground : optional, if true, each image will be normalised with respect to the corresponding 
                                  background image from a stack of background images (one for each shift position) provided in backgroundImgs. 
                                  Defaults to False.
               backgroundImgs   : stack of images, same size as imgs which are used to normalise or for image by image background subtraction. Defaults to None.
               multiBackgrounds : boolean, if True and backgroundImgs is defined, each image will have its own background image subtracted rather than using backgroundImg
               imageScaleFactor : If normToBackground and normToImage are False (default), use this to specify the normalisation factors for each image. Provide a 1D array the same size as the number of shifted images. Each image will be multiplied by the corresponding factor prior to reconstruction. Default is None (i.e. no scaling).
               autoMask         : boolean, mask pixels outside bundle when searching for cores. Defualts to True.
               mask:            : boolean, when reconstructing output image will be masked outside of bundle. Defaults to True

        """
        
        shifts = kwargs.get('shifts', None)
        centreX = kwargs.get('centreX', None)
        centreY = kwargs.get('centreY', None)
        radius = kwargs.get('radius', None)
        filterSize = kwargs.get('filterSize', None)
        normToImage = kwargs.get('normToImage', False)
        normToBackground = kwargs.get('normToBackground', False)
        backgroundImgs = kwargs.get('backgroundImgs', None)
        normalisationImgs = kwargs.get('normalisationImgs', None)
        multiBackgrounds = kwargs.get('multiBackgrounds', False)
        multiNormalisation = kwargs.get('multiNormalisation', False)
        darkFrame = kwargs.get('darkFrame', None)
        singleCalib = kwargs.get('singleCalib', None)
        
        postFilterSize = kwargs.get('postFilterSize', None)
        imageScaleFactor = kwargs.get('imageScaleFactor', None)
        mask = kwargs.get('mask', True)

        if singleCalib is None:
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


        # If a dark frame has been provided, extract core values when no illumination
        if darkFrame is not None:
            darkVals = pybundle.core_values(darkFrame, singleCalib.coreX, singleCalib.coreY, filterSize).astype('double')
        else:
            darkVals = 0


        if normToImage and imgs is not None:
            #print("norm to images")

            imageScaleFactor = np.array(())
            refMean = np.mean(pybundle.core_values(
                imgs[:, :, 0], singleCalib.coreX, singleCalib.coreY, filterSize) - darkVals)

            for idx in range(nImages):
                meanVal = np.mean(pybundle.core_values(
                    imgs[:, :, idx], singleCalib.coreX, singleCalib.coreY, filterSize) - darkVals )
                imageScaleFactor = np.append(
                    imageScaleFactor, refMean / meanVal)

        if normToBackground:
            #print("norm to backgrounds")

            imageScaleFactor = np.array(())

            refMean = np.mean(pybundle.core_values(
                backgroundImgs[:, :, 0], singleCalib.coreX, singleCalib.coreY, filterSize) - darkVals)

            for idx in range(nImages):
                meanVal = np.mean(pybundle.core_values(
                    backgroundImgs[:, :, idx], singleCalib.coreX, singleCalib.coreY, filterSize) - darkVals)
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
        #breakpoint()    

        calib = pybundle.init_tri_interp(calibImg, coreXList, coreYList, centreX, centreY,
                                         radius, gridSize, filterSize=filterSize, background=None, normalise=None)

        # We store the number of cores in a single image
        calib.nCores = np.shape(singleCalib.coreX)[0]
        
        # If we are doing multi-background we need to pull out the core values from each of the
        # background images to use later for image-by-image background subtraction
        if multiBackgrounds:
            
            multiBackgroundVals = np.zeros((calib.nCores, nImages))
            for idx in range(nImages):
                multiBackgroundVals[:, idx] = pybundle.core_values(backgroundImgs[:,:,idx], singleCalib.coreX, singleCalib.coreY, filterSize).astype('double') - darkVals
        
            calib.multiBackgroundVals = multiBackgroundVals
        
        if multiNormalisation:
            
            multiNormalisationVals = np.zeros((calib.nCores, nImages))

            for idx in range(nImages):
                multiNormalisationVals[:, idx] = pybundle.core_values(normalisationImgs[:,:,idx], singleCalib.coreX, singleCalib.coreY, filterSize).astype('double') - darkVals
        
            calib.multiNormalisationVals = multiNormalisationVals

          
        # Have to set 'background' and 'normalise' to None in previous line as the cores are the shifted
        # positions in the full set from all the images and not the actual core position in each images.
        # We later copy across the background/normalise values from the single image calibration.

        calib.coreXShifted = calib.coreX
        calib.coreYShifted = calib.coreY
        

        calib.coreX = singleCalib.coreX
        calib.coreY = singleCalib.coreY

        calib.background = singleCalib.background
        calib.normalise = singleCalib.normalise
        calib.darkVals = darkVals

        # Single calibration 
        calib.backgroundVals = singleCalib.backgroundVals - darkVals
        calib.normaliseVals = singleCalib.normaliseVals - darkVals

        calib.shifts = shifts
        calib.imageScaleFactor = imageScaleFactor
        
        calib.nShifts = nImages

        calib.postFilterSize = postFilterSize
        
        calib.multiBackgrounds = multiBackgrounds
        calib.multiNormalisation = multiNormalisation
        if mask:
            calib.mask = pybundle.get_mask(
                np.zeros((gridSize, gridSize)), (gridSize/2, gridSize/2, gridSize/2))
        else:
            calib.mask = None

        return calib
    

    def recon_multi_tri_interp(imgs, calib, numba = True):
        """ Reconstruct image with super-resolution from set of shifted image. 
        Requires calibration to have been performed and stored in 'calib' as 
        instance of BundleCalibration.
        
        Returns reconstructed image as 2D numpy array
        
        Arguments:
            imgs     : set of shifted images
            calib    : calibration, instance of BundleCalibration (must be
                    created by calib_multi_tri_interp and not calib_tri_interp).
            
        Keyword Arguments:
            numba    : boolean, if True Numba JIT will be used (default).
        """

        nImages = np.shape(imgs)[2]

        cVals = []
        #print("recon filter", str(calib.filterSize))
        for i in range(nImages):

            # Extract intensity from each core
            cValsThis = pybundle.core_values(
                imgs[:, :, i], calib.coreX, calib.coreY, calib.filterSize).astype('double') - calib.darkVals

            if calib.multiBackgrounds:
                #print("subtracting multi background")
                cValsThis = cValsThis - calib.multiBackgroundVals[:,i]
                 
            if calib.multiNormalisation:
                #print("multi normalisation")
                cValsThis = cValsThis / calib.multiNormalisationVals[:,i]
                 

            if calib.imageScaleFactor is not None and calib.multiNormalisation is False:
                #print("Basic image scaling")
                cValsThis = cValsThis * calib.imageScaleFactor[i]
                
                
            if calib.background is not None and calib.multiBackgrounds is False:
                #print("subtracting single background")
                cValsThis = cValsThis - calib.backgroundVals
                
         
            if calib.normalise is not None and calib.multiNormalisation is False:
                #print("normalising")
                cValsThis = cValsThis / calib.normaliseVals

            cVals = np.append(cVals, cValsThis)
               

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


    def get_shifts(imgs, templateSize = None, refSize = None, upsample = 2, **kwargs):
        """ Determines the shift of each image in a stack w.r.t. first image
        
        Return shifts as 2D numpy array.
        
        Arguments:
            
            imgs         : stack of images as 3D numpy array
            
        Keyword Arguments:
            
            templateSize : int, a square of this size is extracted from imgs 
                           as the template, default is 1/4 image size
            refSize      : int, a square of this size is extracted from first 
                           image as the reference image, default is 1/2 image
                           size. Must be bigger than  
                           templateSize and the maximum shift detectable is 
                           (refSize - templateSize)/2   
            upSample     : upsampling factor for images before shift detection  
                           for sub-pixel accuracy, default is 2.
        """

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
    

    def find_shift(img1, img2, templateSize, refSize, upsample, returnMax = False):
        """ Determines shift between two images by Normalised Cross 
        Correlation (NCC). A square template extracted from the centre of img2 
        is compared with a square region extracted from the reference image 
        img1. The size of the template (templateSize) must be less than the 
        size of the reference (refSize). The maximum detectable shift is 
        (refSize - templateSize) / 2.
        
        If returnMax is False, returns shift as a tuple of (x_shift, y_shift).
        If returnMax is True, returns tuple of (shift, cc. peak value).
        
        Arguments:
            img1         : image as 2D numpy array
            img2         : image as 2D numpy array
            templateSize : int, size of square region of img2 to use as template. 
            refSize      : int, size of square region of img1 to template match with
            upsample     : int, factor to scale images by prior to template matching to
                           allow for sub-pixel registration.  
                           
        Keyword Arguments:
            returnMax    : boolean, if true returns cc.peak value as well
                           as shift, default is False. 
                   
        """
        

        if refSize < templateSize or min(np.shape(img1)) < refSize or min(np.shape(img2)) < refSize:
            return -1
        else:

            template = pybundle.extract_central(img2, templateSize).astype('float32')
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
            if returnMax:
                return shift, max_val
            else:
                return shift
            

    def sort_sr_stack(stack, stackLength):
        """ Takes a stack of images and extracts an ordered set
        of images relative to a reference 'blank' frame which is much lower intensity than
        the other frames. 

        The blank frame can be anywhere in the stack, and the output stack will be formed cyclically
        from frames before and after the blank frame. For example, if we have frames:
        
        1  2  3  B  4  5
        
        where B is the blank frame, the function will return a stack in the following order:
        
        4  5  1  2  3
                   
        The input stack, 'stack' should should have (stackLength + 1)  frames to ensure that a 
        and there must be stackLength + 1 images in each cycle
        (i.e. stackLength useful images plus one blank reference image).
        The blank reference image is not returned, i.e the returned stack has stackLength frames.
        
        Input stack should have frame number in third dimension.
        
        Arguments:
            
             stack       : 3D numpy array (y,x, image_number)
             stackLength : number of image after blank frame
             
        """
        
        meanVal = np.mean(np.mean(stack,1),0)        
        blankFrame = np.argmin(meanVal)
        
        outOrder = np.arange(blankFrame + 1, blankFrame + 1 + stackLength)
        
        # Now we allow for images before the blank frame
        outOrderWrapped = np.remainder(outOrder, stackLength + 1)
        
        outStack = np.zeros((np.shape(stack)[0], np.shape(stack)[1], stackLength))                         
        
        for idx in range(stackLength):
            
            outStack[:,:,idx] = stack[:,:,outOrderWrapped[idx]] 
    
        return outStack
    
        
    def multi_tri_backgrounds(calibIn, backgrounds):
         """ Updates a multi_tri calibration with a new set of backgrounds without requiring
         full recalibration.
         
         Returns instance of BundleCalibration.
         
         Arguments:
             calibIn    : bundle calibration, instance of BundleCalibration
             background : background image as 2D numpy array
         """
         calibOut = calibIn
    
         if background is not None:
             calibOut.backgroundVals = pybundle.core_values(background, calibOut.coreX, calibOut.coreY,calibOut.filterSize).astype('double')
             calibOut.background = background
    
         else:
             calibOut.backgroundVals = 0
             calibOut.background = None
             
         return calibOut 

        
    def calib_param_shift(param, images, calibration):
        """ For use when the shifts between the images are linearly dependent on some other parameter. 
        Provide a TRILIN calibration and a 4D stack of images of (x, y, shift, parameter), i.e. an extra 
        dimension to provide examples of shifts for different values of the parameter. The values of the parameter
        corresponding to each set of images is provided in param, i.e. the fourth dimension of images should be
        the same length as param.
        
        Returns a 3D array of calibration factors, giving the gradient and offset of x and y shifts of each image with respect to the parameter.
        
        
        """
        nSets = np.shape(images)[3]
        nShifts = np.shape(images)[2]
        imgRecon = np.zeros((calibration.gridSize, calibration.gridSize, nShifts))
        shifts = np.zeros((nShifts, 2, nSets))
        shiftFit = np.zeros((nShifts, 2, 2))    # 2 dimensions (x,y) and 2 params of linear fit
        assert len(param) == nSets
        for iSet in range(nSets):
            for iShift in range(nShifts):
                imgRecon[:,:, iShift] = pybundle.recon_tri_interp(images[:,:,iShift, iSet], calibration)                
            shifts[:,:,iSet] = SuperRes.get_shifts(imgRecon)
        shifts = shifts * calibration.radius * 2 / calibration.gridSize

        for iShift in range(nShifts):
            shiftFit[iShift, 0, :] = np.polyfit(param, shifts[iShift, 0, :],1)
            shiftFit[iShift, 1, :] = np.polyfit(param, shifts[iShift, 1, :],1)

        return shiftFit    
                
    
        
    def get_param_shift(param, calib):
        """For use when the shifts between the images are linearly dependent on some other parameter. 
        Assuming a prior calibration using calib_param_shift in 'calib', this function returns the 
        current value of the parameter to obtain the image shifts.
        """
        shifts = param * calib[:,:,0] # + calib[:,:,1]
        return shifts
        
    
    def param_calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, shiftsSet, **kwargs):
         """ Performs the calibration step for super-resolution reconstruction multiple times
         for a set of different shifts. All other parameters are the same as for calib_multi_tri_interp.
         """
         calibrationSRs = []
         singleCalib = pyb.calib_multi_ti_interp(calibImg, coreSize, gridSize, **kwargs)
         for idx in range(np.shape(shiftsSet)[2]):
             kwargs.update({'shifts': shiftsSet[:,:,idx]})    # Add this set of shifts a calibrate
             calib = SuperRes.calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, singleCalib = singleCalib, **kwargs)
             calibrationSRs.append(calib)
    
    
class calibrationLUT:   
    
    """ Creates and stores a SR calibration look up table (LUT) containing SR calibrations for different 
    values of some parameter on which the image shift is linearly dependent. paramRange is a tuple of (min, max) values of the parameters, and nCalibrations
    calibration will be generated equally spaced within this range. paramCalib is the output from a 
    calib_param_shift that allows the shifts for a specific.
    """
    def __init__(self, calibImg, imgs, coreSize, gridSize, paramCalib, paramRange, nCalibrations, **kwargs):
        self.paramVals = np.linspace(paramRange[0], paramRange[1], nCalibrations)
        self.calibrations = []
        self.nCalibrations = nCalibrations
        
        # For speed we do the base (single image) calibration here once and then pass this as an argument to calib_multit_tri_interp, otherwise
        # this will get done again every time
        singleCalib = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, **kwargs)

        for idx, paramVal in enumerate(self.paramVals):
            #print("Calibration " + str(idx))
            shift = SuperRes.get_param_shift(paramVal, paramCalib)
            kwargs.update({'shifts': shift})   
            self.calibrations.append(SuperRes.calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, singleCalib = singleCalib, **kwargs))

           
 
    def calibrationSR(self, paramValue): 
        """ Returns the calibration from the LUT which is closest to requested value of the parameters. 
        If paramaeter is outside the range of the paramaters that calibration were performed for, 
        function returns None.
        """

        # If outside range, return None
        if paramValue < self.paramVals[0] or paramValue > self.paramVals[-1]:
            return None

        # If we only have one calibration, and we are not outside the range, then this is the one we want
        if self.nCalibrations == 1:  
            return self.calibrations[0]

       
        idx = round((paramValue - self.paramVals[0]) / (self.paramVals[-1] - self.paramVals[0]) * (self.nCalibrations - 1))
        #print("Desired Value: ", paramValue, "Used Value:", self.paramVals[idx])
        return self.calibrations[idx]
    
    
    def __str__(self):
        return "Calibration LUT of " + str(self.nCalibrations) + " calibrations for param values of " + str(self.paramVals[0]) + " to " + str(self.paramVals[-1])
    
      
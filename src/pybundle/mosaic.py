# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

Mosaic class provides mosaicing functionality.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""
    
import numpy as np
import math
import time

import cv2 as cv

from pybundle import pybundle
from pybundle.utility import extract_central

##############################################################################        
class Mosaic:
    """ The Mosaic class is used for producing mosaics from a sequence of
    images. After instantiating a Mosaic object, use add() to add in images
    and get_mosaic to obtain the current mosaic image.
    
    Arguments:
        mosaicSize    : int, square size of mosaic image
        
    Keyword Arguments:
        resize         : int, images will be resized to a square this size,
                         default is None meaning no resize.
        templateSize   : int, size of square to extract to use as template
                         for shift detection. Default is 1/4 image size.
        refSize        : int, size of square to extract to use as image
                         to compare template to for shift detection. 
                         Default is 1/2 image size.
        cropSize       : int, input images are cropped to a circle of this 
                         diameter before insertion. (default is 
                         0.9 x size of first image added)
        imageType      : str, data type for mosaic, default is the same
                         as first image added
        blend          : boolean, if True (default), images will be added blended,
                         otherwise they are added dead-leaf
        blendDist      : int, distance in pixels from edge of inserted 
                         image to blend with mosaic, default is 40
        minDistforAdd  : int, minimum distance moved before an image
                         will be added to the mosaic, default is 25
        initialX       : int, starting positon of mosaic, default is centre
        initialY       : int, starting positon of mosaic, default is centre  
        boundaryMethod : method to deal with reaching edge: CROP, SCROLL or
                         EXPAND, default is CROP
        expandStep     : int, amount to expand by if EXPAND boundaryMethod
                         is used
        resetThresh    : float, mosaic will reset if correlation peak is
                         below this, default is None (ignore)
        resetIntensity : float, mosaic will reset if mean image value is
                         below this value, default is None (ignore)
        resetSharpness : float, mosaic will reset if image sharpness (mean
                         of gradient) drops below this value, default is None (ignore)                
        
    """
    
    # Constants for method of dealing with reaching the edge of the mosaic image
    CROP = 0
    EXPAND = 1
    SCROLL = 2
    
    # Constants for direction of images reaching the edge of the mosaic image
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3  
    
    lastImageAdded = None
          
    def __init__(self, mosaicSize, **kwargs):        
        
        
        self.mosaicSize = mosaicSize
        self.prevImg = []

        # If None is used for the following
        # sensible values will be selected after the first image
        # is received and __initialise_mosaic() is called
        self.resize = kwargs.get('resize', None)
        self.templateSize = kwargs.get('templateSize', None)
        self.refSize = kwargs.get('refSize', None)
        self.cropSize = kwargs.get('cropSize', None)
        self.imageType = kwargs.get('imageType', None)
        
        # Blending control
        self.blend = kwargs.get('blend', True)
        self.blendDist = kwargs.get('blendDist', 40)
        self.minDistForAdd = kwargs.get('mindDistForAdd', 25)
        
        # Start of mosaic position
        self.initialX = kwargs.get('initialX', round(mosaicSize /  2))
        self.initialY = kwargs.get('initialY', round(mosaicSize /  2))
                
        # Handling of reaching mosaic image edge
        self.boundaryMethod = kwargs.get('boundaryMethod', self.CROP)
        self.expandStep = kwargs.get('expandStep', 50)

        # Detection of mosaicing failure
        self.resetThresh = kwargs.get('resetThresh', None)
        self.resetIntensity = kwargs.get('resetIntensity', None)
        self.resetSharpness = kwargs.get('resetSharpness', None)
        
        # These are created the first time they are needed
        self.mosaic = []
        self.mask = []
        self.blendMask = []
       
        # Initial values
        self.lastShift = [0,0]
        self.lastXAdded = 0
        self.lastYAdded = 0
        self.nImages = 0        
        self.imSize = None  # None tells us to read this from the first image        
        
        self.col = False   # Assume monochrome
        
        return
        
       
    
    # Add image to current mosaic
    def add(self, img):
        """ Add image to current mosaic.
        
        Arguments:
            img    : image as 2D/3D numpy array
        """

        # Before we have first image we can't choose sensible default values, so
        # initialisation is called here if we are on the first image
        if self.nImages == 0:
            self.__initialise_mosaic(img) 
          
                  
        if self.resize is not None: 
            imgResized = cv.resize(img, (self.resize, self.resize))
        else:
            imgResized = img
            
        if self.nImages > 0:
            
            self.lastShift, self.shiftConf = Mosaic.__find_shift(self.prevImg, imgResized, self.templateSize, self.refSize)
           
            # reset mosaic if correlation between two images below threshold
            if self.resetThresh is not None:
                if self.shiftConf < self.resetThresh:
                   
                    self.__initialise_mosaic(img)
                    Mosaic.__insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))

                    return
                
            # reset mosaic if image intensity below threshold
            if self.resetIntensity is not None:
                if np.mean(imgResized) < self.resetIntensity:
                    self.__initialise_mosaic(img)
                    Mosaic.__insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))

                    return
                
            # reset mosaic if image sharpness below threshold  
            if self.resetSharpness is not None:
                refIm = pybundle.extract_central(imgResized, self.refSize)
                
                if refIm.ndim == 3:  # colour
                    gx, gy = np.gradient(refIm, axis = (0,1))
                    gx = np.mean(gx,2)
                    gy = np.mean(gy,2)
                    
                else:                # monochrome
                    gx, gy = np.gradient(refIm)
                   
                gnorm = np.sqrt(gx**2 + gy**2)
                gav = np.mean(gnorm)
                if gav < self.resetSharpness:   
                    self.__initialise_mosaic(img)
                    Mosaic.__insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))
                    return

            self.currentX = self.currentX + self.lastShift[1]
            self.currentY = self.currentY + self.lastShift[0]
            
            distMoved = math.sqrt( (self.currentX - self.lastXAdded)**2 + (self.currentY - self.lastYAdded)**2)
            
            if distMoved >= self.minDistForAdd:
                self.lastXAdded = self.currentX
                self.lastYAdded = self.currentY
                
                for i in range(2):
                    outside, direction, outsideBy = Mosaic.__is_outside_mosaic(self.mosaic, imgResized, (self.currentX, self.currentY))
    
                    if outside == True:
                        if self.boundaryMethod == self.EXPAND: 
                            self.mosaic, self.mosaicWidth, self.mosaicHeight, self.currentX, self.currentY = Mosaic.__expand_mosaic(self.mosaic, max(outsideBy, self.expandStep), direction, self.currentX, self.currentY)
                            outside = False
                        elif self.boundaryMethod == self.SCROLL:
                            self.mosaic, self.currentX, self.currentY = Mosaic.__scroll_mosaic(self.mosaic, outsideBy, direction, self.currentX, self.currentY)
                            outside = False
                    
                if outside == False:
                    if self.blend:
                        Mosaic.__insert_into_mosaic_blended(self.mosaic, imgResized, self.mask, self.blendMask, self.cropSize, self.blendDist, (self.currentX, self.currentY))
                    else:
                        Mosaic.__insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))
                            
        else:  
            # 1st image goes straight into mosaic
            Mosaic.__insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))
            self.lastXAdded = self.currentX
            self.lastYAdded = self.currentY

        self.prevImg = imgResized
        self.nImages = self.nImages + 1



    def get_mosaic(self):
        """ Returns current mosaic image as 2D/3D numpy array
        """
        return self.mosaic
    
        
    def reset(self):
        """ Call to reset mosaic, clearing image. The mosaic will only
        be fully reset once a new image is added. Note that parameters that
        were already initialised will not be reset. 
        """
        self.nImages = 0

    
    def __initialise_mosaic(self, img):
        """ Choose sensible values for non-specified parameters.
        
        Arguments:
            img  : input image to used to choose sensible parameters for 
                    mosaicing, 2D/3D numpy array
        """
        
        if img.ndim == 3:
            self.col = True
            self.nChannels = np.shape(img)[2]
        else:
            self.col = False
        
        if self.imSize is None:
            if self.resize is None:
                self.imSize = min(img.shape[0:2])
            else:
                self.imSize = self.resize
    
        if self.cropSize is None:
            self.cropSize = round(self.imSize * .9)            
        
        if self.templateSize is None:
            self.templateSize = round(self.imSize / 4)
            
        if self.refSize is None:
            self.refSize = round(self.imSize / 2)
            
        if self.imageType is None:
            self.imageType = img.dtype
        
        if np.size(self.mask) == 0:
            self.mask = pybundle.get_mask(np.zeros([self.imSize,self.imSize]),(self.imSize/2,self.imSize/2,self.cropSize / 2))
       
        if self.col:
            self.mosaic = np.zeros((self.mosaicSize, self.mosaicSize, self.nChannels), dtype = self.imageType)
        else:
            self.mosaic = np.zeros((self.mosaicSize, self.mosaicSize), dtype = self.imageType)
    
        self.currentX = self.initialX
        self.currentY = self.initialY
        
        self.nImages = 0
        
        return 
    
    

    @staticmethod    
    def __insert_into_mosaic(mosaic, img, mask, position):
        """ Dead leaf insertion of image into a mosaic at specified position. 
        Only pixels for which mask == 1 are copied.
        
        Arguments:
            mosaic   : current mosaic image, 2D/3D numpy array
            img      : img to insert, 2D/3D numpy array
            mask     : 2D numpy array with values of 1 for pixels to be copied
                       and 0 for pixels not to be copied. Must be same size as img.
            position : position of insertion as tuple of (x,y). This is the
                       pixel the centre of the image will be at.
        
        """
        px = math.floor(position[0] - np.shape(img)[0] / 2)
        py = math.floor(position[1] - np.shape(img)[1] / 2)        
        
        oldRegion = mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]]
        
        oldRegion[np.array(mask)] = img[np.array(mask)]
        mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]] = oldRegion
        
        return
    
    @staticmethod            
    def __insert_into_mosaic_blended(mosaic, img, mask, blendMask, cropSize, blendDist, position):
        """ Insertion of image into a mosaic with cosine window blending. Only pixels from
        image for which mask == 1 are copied. Pixels within blendDist of edge of mosaic
        (i.e. radius of cropSize/2) are blended with existing mosaic pixel values  
        
        Arguments:
           mosaic    : current mosaic image, 2D/3D numpy array
           img       : img to insert, 2D/3D numpy array
           mask      : 2D numpy array with values of 1 for pixels to be copied
                       and 0 for pixels not to be copied. Must be same size as img.           
           blendMask : the cosine window blending mask with weighted pixel values. If passed empty []
                       this will be created
           cropSize  : size of input image.
           blendDist : number which controls the sptial extent of the blending
           position  : position of insertion as tuple of (x,y). This is the
                       pixel the centre of the image will be at.
        
        """
        px = math.floor(position[0] - np.shape(img)[0] / 2)
        py = math.floor(position[1] - np.shape(img)[1] / 2)        
               
        # Region of mosaic we are going to work on
        oldRegion = mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]]

        # Only do blending on non-zero valued pixels of mosaic, otherwise we 
        # blend into nothing
        blendingMask = np.where(oldRegion>0,1,0)

        # If we have a colour image then take the max of the blending mask across
        # the colour channels
        if blendingMask.ndim > 2:
            blendingMask = np.max(blendingMask,2)

        # If first time, create blend mask giving weights to apply for each pixel
        if blendMask == []:
            maskRad = cropSize / 2
            blendImageMask = Mosaic.__cosine_window(np.shape(oldRegion)[0], maskRad, blendDist) 


        imgMask = blendImageMask.copy()
        imgMask[blendingMask == 0] = 1   # For pixels where mosaic == 0 use original pixel values from image 
        imgMask = imgMask * mask
        mosaicMask = 1- imgMask    
        
        if img.ndim > 2:
            mosaicMask = np.expand_dims(mosaicMask, 2)
            imgMask = np.expand_dims(imgMask, 2)

        # Modify region to include blended values from image
        oldRegion = oldRegion * mosaicMask + img * imgMask       

        # Insert it back in
        mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]] = oldRegion
       
        return
   
    @staticmethod
    def __find_shift(img1, img2, templateSize, refSize):
        """ Calculates how far img2 has shifted relative to img1 using
        normalised cross correlation.
        
        Returns tuple of (shift, max_val) where shift is a tuple of (x_shift, y_shift) and 
        max_val is the normalised cross correlation peak value. Returns None if the shift
        cannot be calculated.
           
        Arguments:  
            img1         : reference image as 2D/3D numpy array
            img2         : template image as 2D/3D numpy array
            templateSize : int, a square of this size is extracted from img as 
                           the template
            refSize      : int, a square of this size is extracted from refSize 
                           as the template. 
                           Must be bigger than templateSize and the maximum 
                           shift detectable is (refSize - templateSize)/2
        
        """
        
        if refSize < templateSize or min(np.shape(img1)[0:2]) < refSize or min(np.shape(img2)[0:2]) < refSize:
             return None
        else:
             template = pybundle.extract_central(img2, templateSize)  
             refIm = pybundle.extract_central(img1, refSize)

             res = cv.matchTemplate(pybundle.to8bit(template), pybundle.to8bit(refIm), cv.TM_CCORR_NORMED)
             min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
             shift = [max_loc[0] - (refSize - templateSize), max_loc[1] - (refSize - templateSize)]
             return shift, max_val
            
    
    @staticmethod   
    def __cosine_window(imgSize, circleSize, circleSmooth):
        """ Produce a circular cosine window mask on grid of imgSize * imgSize. Mask
        is 0 for radius > circleSize and 1 for radius < (circleSize - circleSmooth).
        The intermediate region is a smooth cosine function.
        
        Returns mask as 2D numpy array.           
        
        Arguments: 
            imgSize      : int, size of square mask to generate
            circleSize   : int, radius of mask (mask pixels outside here are 0)
            circleSmooth : int, size of smoothing region at the inside edge of the circle. 
                                (mask pixels with a radius less than this are 1)
        """
        
        innerRad = circleSize - circleSmooth
        xM, yM = np.meshgrid(range(imgSize),range(imgSize))
        imgRad = np.sqrt( (xM - imgSize/2) **2 + (yM - imgSize/2) **2)
        mask =  np.cos(math.pi / (2 * circleSmooth) * (imgRad - innerRad))**2
        mask[imgRad < innerRad ] = 1
        mask[imgRad > innerRad + circleSmooth] = 0
        
        return mask    
    
    @staticmethod
    def __is_outside_mosaic(mosaic, img, position):
        """ Checks if position of image to insert into mosaic will result in 
        part of inserted image being outside of mosaic. Returns tuple of 
        boolean (true if outside), side it leaves (using consts defined above)
        and distance is has strayed over the edge. e.g. (True, Mosaic.Top, 20).
        
        Returns tuple of (ouside, side, distance), where outside is True if 
        part of image, side is (one of the) side(s) it has strayed out of, 
        one of Mosaic.Top, Mosaic.Bottom, Mosaic.Left or Mosaic.Right or -1 if 
        outside == False, distance is distance it has strayed outside mosaic in 
        the direction specified in size (0 if outside == False).
        
        Arguments:
            
            mosaic   : mosaic image (strictly can be any numpy array the same size as the mosaic)
            img      : image to be inserted as 2D numpy array
            position : position of insertion as tuple of (x,y). This is the
                       pixel the centre of the image will be at.        
        
            """
        
        imgW = np.shape(img)[0] 
        imgH = np.shape(img)[1] 
        
        mosaicW = np.shape(mosaic)[0]
        mosaicH = np.shape(mosaic)[1]        
                
        left = math.floor(position[0] - imgW / 2)
        top = math.floor(position[1] - imgH / 2)
        
        right = left + imgW 
        bottom = top + imgH 
        
        if left < 0 :
            return True, Mosaic.LEFT, -left
        elif top < 0:
            return True, Mosaic.TOP, -top
        elif right > mosaicW:
            return True, Mosaic.RIGHT, right - mosaicW        
        elif bottom > mosaicH:
            return True, Mosaic.BOTTOM, bottom - mosaicH
        else:
            return False, -1, 0        
     
    @staticmethod
    def __expand_mosaic(mosaic, distance, direction, currentX, currentY):
        """ Increase size of mosaic image by 'distance' in direction 'direction'. Supply
        currentX and currentY position so that these can be modified to be correct
        for new mosaic size. 
        
        Returns tuple of (newMosaic, width, height, newX, newY), where newMosaic 
        is the larger mosaic image as 2D numpy array, width is the x-size of 
        the new mosaic, height is the y-size of the new mosaic, newX is the x 
        position of the last image insertion in the new mosaic, newY is the y 
        position of the last image insertion in the new mosaic.
        
        Arguments:            
            mosaic    : input mosaic image as 2D numpy array
            distance  : pixels to expand by
            direction : side to expand, one of Mosaic.Top, Mosaic.Bottom, 
                         Mosaic.Left or Mosaic.Right
            currentX  : x position of last image insertion into mosaic
            currentY  : y position of last image insertion into mosaic
        
        """          
        mosaicWidth = np.shape(mosaic)[0]
        mosaicHeight = np.shape(mosaic)[1]
        if mosaic.ndim > 2:
            mosaicChannels = np.shape(mosaic)[2]
        else:
            mosaicChannels = 1

        if direction == Mosaic.LEFT:
            newMosaicWidth = mosaicWidth + distance
            newMosaic = np.squeeze(np.zeros((newMosaicWidth, mosaicHeight, mosaicChannels), mosaic.dtype))
            newMosaic[distance:distance + mosaicWidth,:] = mosaic
            return newMosaic, newMosaicWidth, mosaicHeight, currentX + distance, currentY
             
        if direction == Mosaic.TOP:
            newMosaicHeight = mosaicHeight + distance
            newMosaic = np.squeeze(np.zeros((mosaicWidth, newMosaicHeight, mosaicChannels), mosaic.dtype))
            newMosaic[:,distance:distance + mosaicHeight] = mosaic
            return newMosaic, mosaicWidth, newMosaicHeight, currentX,  currentY + distance 
        
        if direction == Mosaic.RIGHT:
            newMosaicWidth = mosaicWidth + distance
            newMosaic = np.squeeze(np.zeros((newMosaicWidth, mosaicHeight, mosaicChannels), mosaic.dtype))
            newMosaic[0: mosaicWidth,:] = mosaic
            return newMosaic, newMosaicWidth, mosaicHeight, currentX, currentY
        
        if direction == Mosaic.BOTTOM:
            newMosaicHeight = mosaicHeight + distance
            newMosaic = np.squeeze(np.zeros((mosaicWidth, newMosaicHeight, mosaicChannels), mosaic.dtype))
            newMosaic[:, 0:mosaicHeight ] = mosaic
            return newMosaic, mosaicWidth, newMosaicHeight,  currentX , currentY 
        
    @staticmethod
    def __scroll_mosaic(mosaic, distance, direction, currentX, currentY):
        """ Scroll mosaic to allow mosaicing to continue past edge of mosaic. Pixel 
        values will be lost. Supply currentX and currentY position so that these
        can be modified to be correct for new mosaic size.
        
        Return: tuple of (newMosaic, width, height, newX, newY), where 
        newMosaic is the larger mosaic image as 2D numpy array, width is the 
        x-size of the new mosaic, height is the y-size of the new mosaic, newX 
        is the x position of the last image insertion in the new mosaic, newY 
        is the y position of the last image insertion in the new mosaic.
        
        Arguments:
            mosaic    : input mosaic image as 2D numpy array
            distance  : pixels to expand by
            direction : side to expand, one of Mosaic.Top, Mosaic.Bottom, 
                        Mosaic.Left or Mosaic.Right
            currentX  : x position of last image insertion into mosaic
            currentY  : y position of last image insertion into mosaic        
        
        """
        mosaicWidth = np.shape(mosaic)[0]
        mosaicHeight = np.shape(mosaic)[1]

        if direction == Mosaic.LEFT:       
            newMosaic = np.roll(mosaic,distance,0)
            newMosaic[0:distance:,:] = 0 
            return newMosaic, currentX + distance, currentY
             
        if direction == Mosaic.TOP:
            newMosaic = np.roll(mosaic,distance,1)
            newMosaic[:, 0:distance] = 0 
            return newMosaic, currentX,  currentY + distance 
        
        if direction == Mosaic.RIGHT:
            newMosaic = np.roll(mosaic,-distance,0)
            newMosaic[-distance:,:] = 0 
            return newMosaic, currentX  - distance, currentY
        
        if direction == Mosaic.BOTTOM:
            newMosaic = np.roll(mosaic,-distance,1)
            newMosaic[:, -distance:] = 0 
            return newMosaic, currentX,  currentY   - distance
        
        
     
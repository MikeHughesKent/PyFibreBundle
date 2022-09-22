# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.


Mosaic class provides mosaicing functionality.


@author: Mike Hughes
Applied Optics Group, University of Kent
https://github.com/mikehugheskent
"""
    
import numpy as np
import math
import time
import cv2 as cv
from pybundle import pybundle

##############################################################################        
class Mosaic:
    
    # Constants for method of dealing with reaching the edge of the mosaic image
    CROP = 0
    EXPAND = 1
    SCROLL = 2
    
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
        # is received
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
        self.imSize = None  # -1 tell us to read this from the first image
        
      
   
        
        return
       
        
       
    def initialise(self, img):
        
        # Choose sensible values for non-specified parameters
        
        if self.imSize is None:
            if self.resize is None:
                self.imSize = min(img.shape)
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
       
        self.mosaic = np.zeros((self.mosaicSize, self.mosaicSize), dtype = self.imageType)

        self.currentX = self.initialX
        self.currentY = self.initialY
        
        self.nImages = 0
        
        return 
    
    
    # Add image to current mosaic
    def add(self, img):

        # Before we have first image we can't choose sensible default values, so
        # initialisation is called here if we are on the first image
        if self.nImages == 0:
            self.initialise(img) 

                  
        if self.resize is not None: 
            imgResized = cv.resize(img, (self.resize, self.resize))
        else:
            imgResized = img
            
        if self.nImages > 0:
            
            self.lastShift, self.shiftConf = Mosaic.find_shift(self.prevImg, imgResized, self.templateSize, self.refSize)
           
            if self.resetThresh is not None:
                if self.shiftConf < self.resetThresh:
                    self.initialise(img)
                    Mosaic.insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))

                    return
            
            if self.resetIntensity is not None:
                if np.mean(imgResized) < self.resetIntensity:
                    self.initialise(img)
                    Mosaic.insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))

                    return
                
            if self.resetSharpness is not None:
                refIm = pybundle.extract_central(imgResized, self.refSize)

                gx, gy = np.gradient(refIm)
                gnorm = np.sqrt(gx**2 + gy**2)
                gav = np.mean(gnorm)
                if gav < self.resetSharpness:   
                    self.initialise(img)
                    Mosaic.insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))
                    return

            self.currentX = self.currentX + self.lastShift[1]
            self.currentY = self.currentY + self.lastShift[0]
            
            distMoved = math.sqrt( (self.currentX - self.lastXAdded)**2 + (self.currentY - self.lastYAdded)**2)
            self.minDistForAdd = 0
            if distMoved >= self.minDistForAdd:
                self.lastXAdded = self.currentX
                self.lastYAdded = self.currentY
                
                for i in range(2):
                    outside, direction, outsideBy = Mosaic.is_outside_mosaic(self.mosaic, imgResized, (self.currentX, self.currentY))
    
                    if outside == True:
                        if self.boundaryMethod == self.EXPAND: 
                            self.mosaic, self.mosaicWidth, self.mosaicHeight, self.currentX, self.currentY = Mosaic.expand_mosaic(self.mosaic, max(outsideBy, self.expandStep), direction, self.currentX, self.currentY)
                            outside = False
                        elif self.boundaryMethod == self.SCROLL:
                            self.mosaic, self.currentX, self.currentY = Mosaic.scroll_mosaic(self.mosaic, outsideBy, direction, self.currentX, self.currentY)
                            outside = False
                    
                if outside == False:
                    if self.blend:
                        #t1 = time.perf_counter()
                        Mosaic.insert_into_mosaic_blended(self.mosaic, imgResized, self.mask, self.blendMask, self.cropSize, self.blendDist, (self.currentX, self.currentY))
                        #print("Time add to mosaic:", str(time.perf_counter() -t1))

                    else:
                        Mosaic.insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))
                            
        else:  
            # 1st image goes straight into mosaic
            Mosaic.insert_into_mosaic(self.mosaic, imgResized, self.mask, (self.currentX, self.currentY))
            self.lastXAdded = self.currentX
            self.lastYAdded = self.currentY

        self.prevImg = imgResized
        self.nImages = self.nImages + 1




    # Return mosaic image
    def get_mosaic(self):
        return self.mosaic
        


    def reset(self):
        self.nImages = 0
            
        
    # Dead leaf insertion of image into a mosaic at position. Only pixels for
    # which mask == 1 are copied
    def insert_into_mosaic(mosaic, img, mask, position):
        
        px = math.floor(position[0] - np.shape(img)[0] / 2)
        py = math.floor(position[1] - np.shape(img)[1] / 2)
        
        
        oldRegion = mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]]
        oldRegion[np.array(mask)] = img[np.array(mask)]
        mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]] = oldRegion
        return
    
    
    
    
    # Insertion of image into a mosaic with cosine window blending. Only pixels from
    # image for which mask == 1 are copied. Pixels within blendDist of edge of mosaic
    # (i.e. radius of cropSize/2) are blended with existing mosaic pixel values
    def insert_into_mosaic_blended(mosaic, img, mask, blendMask, cropSize, blendDist, position):
        
        
        px = math.floor(position[0] - np.shape(img)[0] / 2)
        py = math.floor(position[1] - np.shape(img)[1] / 2)
        
               
        # Region of mosaic we are going to work on
        oldRegion = mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]]

        # Only do blending on non-zero valued pixels of mosaic, otherwise we 
        # blend into nothing
        blendingMask = np.where(oldRegion>0,1,0)

        # If first time, create blend mask giving weights to apply for each pixel
        if blendMask == []:
            maskRad = cropSize / 2
            blendImageMask = Mosaic.cosine_window(np.shape(oldRegion)[0], maskRad, blendDist) 


        imgMask = blendImageMask.copy()
        imgMask[blendingMask == 0] = 1   # For pixels where mosaic == 0 use original pixel values from image 
        imgMask = imgMask * mask
        mosaicMask = 1- imgMask          

        # Modify region to include blended values from image
        oldRegion = oldRegion * mosaicMask + img * imgMask       

        # Insert it back in
        mosaic[px:px + np.shape(img)[0] , py :py + np.shape(img)[1]] = oldRegion
       
        return
    
      
    
   
    
    # Calculates how far img2 has shifted relative to img1
    def find_shift(img1, img2, templateSize, refSize):
    
        if refSize < templateSize or min(np.shape(img1)) < refSize or min(np.shape(img2)) < refSize:
             return -1
        else:
             template = pybundle.extract_central(img2, templateSize)  
             refIm = pybundle.extract_central(img1, refSize)

             res = cv.matchTemplate(pybundle.to8bit(template), pybundle.to8bit(refIm), cv.TM_CCORR_NORMED)
             min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
             shift = [max_loc[0] - (refSize - templateSize), max_loc[1] - (refSize - templateSize)]
             #shift = 0
             return shift, max_val
                
         
            
    
    # Extracts sqaure of size boxSize from centre of img
    def extract_central(img, boxSize):
        w = np.shape(img)[0]
        h = np.shape(img)[1]

        cx = w/2
        cy = h/2
        boxSemiSize = min(cx,cy,boxSize)
        
        img = img[math.floor(cx - boxSemiSize):math.floor(cx + boxSemiSize), math.ceil(cy- boxSemiSize): math.ceil(cy + boxSemiSize)]
        return img
    
    
    
        
    # Produce a circular cosine window mask on grid of imgSize * imgSize. Mask
    # is 0 for radius > circleSize and 1 for radius < (circleSize - cicleSmooth)
    # The intermediate region is a smooth cosine function.
    def cosine_window(imgSize, circleSize, circleSmooth):
        
        innerRad = circleSize - circleSmooth
        xM, yM = np.meshgrid(range(imgSize),range(imgSize))
        imgRad = np.sqrt( (xM - imgSize/2) **2 + (yM - imgSize/2) **2)
        mask =  np.cos(math.pi / (2 * circleSmooth) * (imgRad - innerRad))**2
        mask[imgRad < innerRad ] = 1
        mask[imgRad > innerRad + circleSmooth] = 0
        return mask
    
    
    
    
    # Checks if position of image to insert into mosaic will result in 
    # part of inserted images being outside of mosaic. Returns tuple of 
    # boolean (true if outside), side it leaves (using consts defined above)
    # and distance is has strayed over the edge. e.g. (True, Mosaic.Top, 20).
    def is_outside_mosaic(mosaic, img, position):
        
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
        
     
    # Increase size of mosaic image by 'distance' in direction 'direction'. Supply
    # currentX and currentY position so that these can be modified to be correct
    # for new mosaic size
    def expand_mosaic(mosaic, distance, direction, currentX, currentY):
        mosaicWidth = np.shape(mosaic)[0]
        mosaicHeight = np.shape(mosaic)[1]

        if direction == Mosaic.LEFT:
            newMosaicWidth = mosaicWidth + distance
            newMosaic = np.zeros((newMosaicWidth, mosaicHeight), mosaic.dtype)
            newMosaic[distance:distance + mosaicWidth,:] = mosaic
            return newMosaic, newMosaicWidth, mosaicHeight, currentX + distance, currentY
             
        if direction == Mosaic.TOP:
            newMosaicHeight = mosaicHeight + distance
            newMosaic = np.zeros((mosaicWidth, newMosaicHeight), mosaic.dtype)
            newMosaic[:,distance:distance + mosaicHeight] = mosaic
            return newMosaic, mosaicWidth, newMosaicHeight, currentX,  currentY + distance 
        
        if direction == Mosaic.RIGHT:
            newMosaicWidth = mosaicWidth + distance
            newMosaic = np.zeros((newMosaicWidth, mosaicHeight), mosaic.dtype)
            newMosaic[0: mosaicWidth,:] = mosaic
            return newMosaic, newMosaicWidth, mosaicHeight, currentX, currentY
        
        if direction == Mosaic.BOTTOM:
            newMosaicHeight = mosaicHeight + distance
            newMosaic = np.zeros((mosaicWidth, newMosaicHeight), mosaic.dtype)
            newMosaic[:, 0:mosaicHeight ] = mosaic
            return newMosaic, mosaicWidth, newMosaicHeight,  currentX , currentY 
        
        
        
    # Scroll mosaic to allow mosaicing to continue past edge of mosaic. Pixel 
    # values will be lost. Supply currentX and currentY position so that these
    # can be modified to be correct for new mosaic size
    def scroll_mosaic(mosaic, distance, direction, currentX, currentY):
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
        
        
     
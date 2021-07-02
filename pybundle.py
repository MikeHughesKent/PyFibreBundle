# -*- coding: utf-8 -*-
"""
PyBundle is an open source Python package for image processing of
fibre bundle images.

@author: Mike Hughes
"""

import cv2 as cv
import numpy as np
from skimage.transform import hough_circle, hough_circle_peaks
import math

class PyBundle:
    
   
    
    def __init__(self):
        pass
       

    # Applies Gaussian filter to image
    def gFilter(img, filterSize):
        
        kernelSize = round(filterSize * 6)           # Kernal size needs to be larger than sigma
        kernelSize = kernelSize + 1 - kernelSize%2   # Kernel size must be odd
        imgFilt = cv.GaussianBlur(img,(kernelSize,kernelSize), filterSize)
        return imgFilt
    
        
    # *** Locates bundle in an image by thresholding and searching for largest
    # connected region. Returns tuple of (centreX, centreY, radius)
    def findBundle(img, **kwargs):
        
        filterSize = kwargs.get('filterSize', 4)
        
        # Filter to minimise effects of structure in bundle
        kernelSize = round(filterSize * 6)           # Kernal size needs to be larger than sigma
        kernelSize = kernelSize + 1 - kernelSize%2   # Kernel size must be odd
        imgFilt = cv.GaussianBlur(img,(kernelSize,kernelSize), filterSize)
        
        # Threshold to binary and then look for connected regions
        thres, imgBinary = cv.threshold(imgFilt,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)
        num_labels, labels, stats, centroid  = cv.connectedComponentsWithStats(imgBinary, 8, cv.CV_32S)
        
        # Region 0 is background, so find largest of other regions
        sizes = stats[1:,4]
        biggestRegion = sizes.argmax() + 1
        
        # Find distance from centre to each edge and take minimum as safe value for radius
        centreX = round(centroid[biggestRegion,0]) 
        centreY = round(centroid[biggestRegion,1])
        radius1 = centroid[biggestRegion,0] - stats[biggestRegion,0]
        radius2 = centroid[biggestRegion,1] - stats[biggestRegion,1]
        radius3 = -(centroid[biggestRegion,0] - stats[biggestRegion,2]) + stats[biggestRegion,0]
        radius4 = -(centroid[biggestRegion,1] - stats[biggestRegion,3]) + stats[biggestRegion,1]
        radius = round(min(radius1, radius2, radius3, radius4))          
              
        return centreX, centreY, radius
    
        
    # Extracts a square around the bundle using specified co-ordinates  
    def cropRect(img,loc):
        cx = loc[0]
        cy = loc[1]
        rad = loc[2]
        imgCrop = img[cy-rad:cy+ rad, cx-rad:cx+rad]
        
        # Correct the co-ordinates of the bundles so that they
        # are correct for new cropped image
        newLoc = [rad,rad,loc[2] ]
   
        return imgCrop, newLoc
    
        
    # Finds the mask and applies it to set all values outside
    def maskAuto(img, loc):
        imgMasked = np.multiply(img, PyBundle.getMask(img,loc))
        return imgMasked
    
    # Sets all pixels outside bundle to 0
    def mask(img, mask):
        imgMasked = np.multiply(img, mask)
        return imgMasked
    
    # Returns location of bundle (tuple of centreX, centreY, radius) and a
    # a mask image with all pixel inside of bundle = 1, and those outside = 2
    def bundleLocate(img):
        loc = PyBundle.findBundle(img)
        imgD, croppedLoc = PyBundle.cropRect(img,loc)
        mask = PyBundle.getMask(imgD, croppedLoc)
        return loc, mask
    
    # Sequentially crops image to bundle, applies Gaussian filter and then
    # sets pixels outside bundle to 0
    def cropFilterMask(img, loc, mask, filterSize):
        
        img, newLoc = PyBundle.cropRect(img, loc)
        img = PyBundle.gFilter(img, filterSize)
        img = PyBundle.mask(img, mask)
        return img


    # Returns a mask, 1 inside bundle, 0 outside bundle
    def getMask(img, loc):
        cx = loc[0]
        cy = loc[1]
        rad = loc[2]
        mY,mX = np.meshgrid(range(img.shape[0]),range(img.shape[1]))
        
        m = np.square(mX - cx) +  np.square(mY - cy)   
        imgMask = np.transpose(m < rad**2)
         
        return imgMask
    
    
    # Find cores in bundle image using Hough transform. This generally
    # does not work well as findCores and is a lot slower
    def findCoresHough(img, **kwargs):
       
        scaleFac = kwargs.get('scaleFactor', 2)
        cannyLow = kwargs.get('cannyLow', .05)
        cannyHigh = kwargs.get('cannyHigh', .8)
        estRad = kwargs.get('estRad', 1)
        minRad = kwargs.get('minRad', np.floor(max(1,estRad)).astype('int'))
        maxRad = kwargs.get('maxRad', np.floor(minRad + 2).astype('int'))
        minSep = kwargs.get('minSep', estRad * 2)
        darkRemove = kwargs.get('darkRemove', 2)
        gFilterSize = kwargs.get('filterSize', estRad / 2)

        
        imgR = cv.resize(img, [scaleFac * np.size(img,0),  scaleFac * np.size(img,1)] ).astype(float)
              
        # Pre filter with Gaussian and Canny
        imgF = PyBundle.gFilter(imgR, gFilterSize*scaleFac) 
        imgF = imgF.astype('uint8')
        edges = cv.Canny(imgF,cannyLow,cannyHigh)
        
        # Using Scikit-Image Hough implementation, trouble getting CV to work
        radii = range(math.floor(minRad * scaleFac),math.ceil(maxRad * scaleFac))
        circs = hough_circle(edges, radii, normalize=True, full_output=False)
 
 
        minSepScaled = np.round(minSep * scaleFac).astype('int')
       
        for i in range(np.size(circs,0)):
            circs[i,:,:] = np.multiply(circs[i,:,:], imgF) 
        
        #circs = np.multiply(circs, imgF)
        
        accums, cx, cy, radii = hough_circle_peaks(circs, radii, min_xdistance = minSepScaled, min_ydistance = minSepScaled)

        # Remove any finds that lie on dark points
        meanVal = np.mean(imgF[cy,cx])
        stdVal = np.std(imgF[cy,cx])
        removeCore = np.zeros_like(cx)
        for i in range(np.size(cx)):
            if imgF[cy[i],cx[i]] < meanVal - darkRemove * stdVal :
                removeCore[i] = 1
        cx = cx[removeCore !=1]        
        cy = cy[removeCore !=1]      

        cx = cx / scaleFac    
        cy = cy / scaleFac     
            
        
        return cx,cy, imgF, edges, circs
    
      
    

    # Find cores in bundle image using regional maxima. Generally fast and 
    #accurate
    def findCores(img, coreSpacing):
       
        
        # Pre-filtering helps to minimse noise and reduce efffect of
        # multimodal patterns
        imgF = PyBundle.gFilter(img, coreSpacing/5)
       
        # Find regional maximum by taking difference between dilated and original
        # image. Because of the way dilation works, the local maxima are not changed
        # and so these will have a value of 0
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(coreSpacing,coreSpacing))
        imgD = cv.dilate(imgF, kernel)
        imgMax = 255 - (imgF - imgD)  # we need to invert the image

        # Just keep the maxima
        thres, imgBinary = cv.threshold(imgMax,0,1,cv.THRESH_BINARY+cv.THRESH_OTSU)

        # Dilation step helps deal with mode patterns which have led to multiple
        # maxima within a core, the two maxima will end up merged into one connected
        # region
        elSize = math.ceil(coreSpacing / 3)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(elSize,elSize))
        imgDil = cv.dilate(imgBinary, kernel)

        # Core centres are centroids of connected regions
        nReg, p1, p2, centroid = cv.connectedComponentsWithStats(imgDil, 8, cv.CV_32S)
        cx = centroid[1:,0]  # The 1st entry is the background
        cy = centroid[1:,1]
                
        return cx, cy
    
    
    
    
    
    
    
      
        
    
 
        



# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.


BundleCalibration class stores the results of calibration for triangualar
linear interpretation.


@author: Mike Hughes, Applied Optics Group, University of Kent
"""

class BundleCalibration:
    
            
    coreX = None
    coreY = None
    coreXInitial = None
    coreYInital = None
    gridSize = None
    mapping = None
    tri = None
    filterSize = None
    normalise = None
    mapping = None
    baryCoords = None
    normaliseVals = None
    background = None
    backgroundVals = None
    radius = None
    col = False
    
    def __init__(self):
        return
    
    def __str__(self):
        
        if self.coreX is not None:
            nCores = len(self.coreX)
        else:
            nCores = 0
            
        return "Bundle Interpolation Calibration: " + str(nCores) + " found."
        
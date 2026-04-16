# -*- coding: utf-8 -*-
"""
Tests sim functions of PyFibreBundle.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import unittest 


import numpy as np

from pybundle.sim import predict_num_cores, predict_radius, predict_spacing, core_centres
 
class TestSim(unittest.TestCase):

    
    def test_predict_num_cores(self):
        
        coreX, coreY = core_centres(spacing = 3, radius = 300, shape='circle')
        num_predicted = predict_num_cores(spacing = 2, radius = 200, shape = 'circle')
        self.assertAlmostEqual(np.shape(coreX)[0], num_predicted, delta = num_predicted / 50)
        
        coreX, coreY = core_centres(spacing = 3, radius = 300, shape='square')
        num_predicted = predict_num_cores(spacing = 2, radius = 200, shape = 'square')
        self.assertAlmostEqual(np.shape(coreX)[0], num_predicted, delta = num_predicted / 50)


    def test_predict_radius(self):
        
        radius_predicted = predict_radius(spacing = 3, numCores = 30000, shape = 'circle')
        coreX, coreY = core_centres(spacing = 3, radius = radius_predicted, shape='circle')
        self.assertAlmostEqual(np.shape(coreX)[0], 30000, delta = 1000)
        
    
    def test_predict_spacing(self):
        
        spacing_predicted = predict_spacing(radius = 400, numCores = 30000, shape = 'circle')
        coreX, coreY = core_centres(spacing = spacing_predicted, radius = 400, shape='circle')
        self.assertAlmostEqual(np.shape(coreX)[0], 30000, delta = 1000)
        
    


if __name__ == '__main__':
    unittest.main()
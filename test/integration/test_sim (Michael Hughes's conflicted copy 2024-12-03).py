# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:40:10 2024

@author: AOG

"""
import matplotlib.pyplot as plt

import context

import pybundle.sim

from pybundle.sim import BundleSim
bundle = BundleSim()
bundle.generate()

plt.imshow(bundle.bundle_image)

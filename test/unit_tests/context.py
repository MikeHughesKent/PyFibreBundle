# -*- coding: utf-8 -*-
"""
Path setup for unit tests.

@author: Mike Hughes, Applied Optics Group, University of Kent
"""

import sys
import os

testdir = os.path.dirname(__file__)
srcdir = '../../src/'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

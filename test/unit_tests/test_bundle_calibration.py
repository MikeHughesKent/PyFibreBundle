# -*- coding: utf-8 -*-
"""
Unit tests for pybundle.bundle_calibration.BundleCalibration class.

Tests cover:
    - Default attribute values after construction
    - __str__ output
    - Attribute assignment

"""

import unittest
import numpy as np

import context

from pybundle.bundle_calibration import BundleCalibration


class TestBundleCalibrationStr(unittest.TestCase):

    def test_str_with_no_cores(self):
        """__str__ when coreX is None should report 0 cores."""
        calib = BundleCalibration()
        s = str(calib)
        self.assertIn('0', s)

    def test_str_with_cores_reports_count(self):
        """__str__ with coreX set should report the correct number of cores."""
        calib = BundleCalibration()
        calib.coreX = np.arange(500)
        s = str(calib)
        self.assertIn('500', s)

    def test_str_returns_string(self):
        calib = BundleCalibration()
        self.assertIsInstance(str(calib), str)


class TestBundleCalibrationAttributeAssignment(unittest.TestCase):
    """Attributes can be set after construction."""

    def test_assign_core_positions(self):
        calib = BundleCalibration()
        calib.coreX = np.array([10.0, 20.0, 30.0])
        calib.coreY = np.array([15.0, 25.0, 35.0])
        self.assertEqual(len(calib.coreX), 3)
        self.assertEqual(len(calib.coreY), 3)

    def test_assign_grid_size(self):
        calib = BundleCalibration()
        calib.gridSize = 512
        self.assertEqual(calib.gridSize, 512)

    def test_assign_radius(self):
        calib = BundleCalibration()
        calib.radius = 200.0
        self.assertAlmostEqual(calib.radius, 200.0)

    def test_assign_col_flag(self):
        calib = BundleCalibration()
        calib.col = True
        self.assertTrue(calib.col)

    def test_multiple_instances_independent(self):
        """Two calibration instances should not share state."""
        c1 = BundleCalibration()
        c2 = BundleCalibration()
        c1.gridSize = 256
        c2.gridSize = 512
        self.assertEqual(c1.gridSize, 256)
        self.assertEqual(c2.gridSize, 512)


if __name__ == '__main__':
    unittest.main()

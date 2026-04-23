# -*- coding: utf-8 -*-
"""
Unit tests for pybundle.sim module.
"""

import unittest
import numpy as np

import context

from pybundle.sim import predict_num_cores, predict_bundle_radius, predict_core_spacing, regular_core_lattice

class TestRegularCoreLattice(unittest.TestCase):

    def test_returns_equal_length_1d_arrays(self):
        """ core x and core y must be same length arrays"""
        coreX, coreY = regular_core_lattice(spacing=3, radius=30)
        self.assertIsInstance(coreX, np.ndarray)
        self.assertIsInstance(coreY, np.ndarray)
        self.assertEqual(coreX.ndim, 1)
        self.assertEqual(coreY.ndim, 1)
        self.assertEqual(len(coreX), len(coreY))

        coreX, coreY = regular_core_lattice(spacing=3, radius=30, packing='square')
        self.assertIsInstance(coreX, np.ndarray)
        self.assertIsInstance(coreY, np.ndarray)
        self.assertEqual(coreX.ndim, 1)
        self.assertEqual(coreY.ndim, 1)
        self.assertEqual(len(coreX), len(coreY))

        coreX, coreY = regular_core_lattice(spacing=3, radius=30, shape='square')
        self.assertIsInstance(coreX, np.ndarray)
        self.assertIsInstance(coreY, np.ndarray)
        self.assertEqual(coreX.ndim, 1)
        self.assertEqual(coreY.ndim, 1)
        self.assertEqual(len(coreX), len(coreY))

    def test_num_cores_matches_prediction_for_square_shape_hex(self):
        """ Allows 5% error, just to check this is essentially producing the
        right size output."""
        spacing = 3
        radius = 90
        coreX, coreY = regular_core_lattice(spacing=spacing, radius=radius, shape='square', packing='hex')
        num_predicted = predict_num_cores(spacing=spacing, radius=radius, shape='square', packing='hex')
        self.assertEqual(len(coreX), len(coreY))
        self.assertAlmostEqual(len(coreX), num_predicted, delta=num_predicted * 0.05)

    def test_num_cores_matches_prediction_for_square_shape_square(self):
        """ Allows 5% error, just to check this is essentially producing the
        right size output."""
        spacing = 3
        radius = 90
        coreX, coreY = regular_core_lattice(spacing=spacing, radius=radius, shape='square', packing='square')
        num_predicted = predict_num_cores(spacing=spacing, radius=radius, shape='square', packing='square')

        self.assertEqual(len(coreX), len(coreY))
        self.assertAlmostEqual(len(coreX), num_predicted, delta=num_predicted * 0.05)

    def test_num_cores_matches_prediction_for_circle_shape_hex(self):
        """ Allows 5% error, just to check this is essentially producing the
        right size output."""
        spacing = 3
        radius = 90
        coreX, coreY = regular_core_lattice(spacing=spacing, radius=radius, shape='circle', packing='hex')
        num_predicted = predict_num_cores(spacing=spacing, radius=radius, shape='circle', packing='hex')
        self.assertEqual(len(coreX), len(coreY))
        self.assertAlmostEqual(len(coreX), num_predicted, delta=num_predicted * 0.05)

    def test_num_cores_matches_prediction_for_circle_shape_square(self):
        """ Allows 5% error, just to check this is essentially producing the
        right size output."""
        spacing = 3
        radius = 90
        coreX, coreY = regular_core_lattice(spacing=spacing, radius=radius, shape='circle', packing='square')
        num_predicted = predict_num_cores(spacing=spacing, radius=radius, shape='circle', packing='square')
        self.assertEqual(len(coreX), len(coreY))
        self.assertAlmostEqual(len(coreX), num_predicted, delta=num_predicted * 0.05)



class TestPredictBundleRadius(unittest.TestCase):

    def test_returns_number(self):
        """predict_bundle_radius should return a numeric value."""
        result = predict_bundle_radius(spacing=3, num_cores=1000)
        self.assertIsInstance(result, (int, float))
        self.assertGreater(result, 0)

    def test_all_shape_packing_combinations(self):
        """Test all valid combinations of shape and packing."""
        shapes = ['circle', 'square']
        packings = ['hex', 'square']
        for shape in shapes:
            for packing in packings:
                with self.subTest(shape=shape, packing=packing):
                    result = predict_bundle_radius(spacing=3, num_cores=1000, shape=shape, packing=packing)
                    self.assertGreater(result, 0)

    def test_larger_spacing_requires_larger_radius(self):
        """Doubling the spacing should roughly double the required radius."""
        radius_s1 = predict_bundle_radius(spacing=1, num_cores=1000)
        radius_s2 = predict_bundle_radius(spacing=2, num_cores=1000)
        self.assertGreater(radius_s2, radius_s1)
        self.assertAlmostEqual(radius_s2, 2 * radius_s1, delta=0.01)

    def test_more_cores_requires_larger_radius(self):
        """More cores should require a larger radius."""
        radius_n1 = predict_bundle_radius(spacing=1, num_cores=1000)
        radius_n2 = predict_bundle_radius(spacing=1, num_cores=10000)
        self.assertGreater(radius_n2, radius_n1)

    def test_circle_larger_than_square_for_same_cores(self):
        """Circle shape requires larger radius than square for same core count."""
        radius_circle = predict_bundle_radius(spacing=1, num_cores=10000, shape='circle', packing='hex')
        radius_square = predict_bundle_radius(spacing=1, num_cores=10000, shape='square', packing='hex')
        self.assertLess(radius_square, radius_circle)

    def test_predicted_radius_with_lattice_generation(self):
        """Use predict_bundle_radius to generate a lattice and verify core count."""
        spacing = 2.5
        target_cores = 5000
        predicted_radius = predict_bundle_radius(spacing=spacing, num_cores=target_cores, shape='circle', packing='hex')
        coreX, coreY = regular_core_lattice(spacing=spacing, radius=predicted_radius, shape='circle', packing='hex')
        # Allow 10% error since prediction is approximate
        self.assertAlmostEqual(len(coreX), target_cores, delta=target_cores * 0.1)


class TestPredictCoreSpacing(unittest.TestCase):

    def test_returns_number(self):
        """predict_core_spacing should return a numeric value."""
        result = predict_core_spacing(radius=50, num_cores=1000)
        self.assertIsInstance(result, (int, float))
        self.assertGreater(result, 0)

    def test_all_shape_packing_combinations(self):
        """Test all valid combinations of shape and packing."""
        shapes = ['circle', 'square']
        packings = ['hex', 'square']
        for shape in shapes:
            for packing in packings:
                with self.subTest(shape=shape, packing=packing):
                    result = predict_core_spacing(radius=50, num_cores=1000, shape=shape, packing=packing)
                    self.assertGreater(result, 0)

    def test_larger_radius_allows_larger_spacing(self):
        """Doubling the radius should roughly double the allowed spacing."""
        spacing_r1 = predict_core_spacing(radius=50, num_cores=1000)
        spacing_r2 = predict_core_spacing(radius=100, num_cores=1000)
        self.assertGreater(spacing_r2, spacing_r1)
        self.assertAlmostEqual(spacing_r2, 2 * spacing_r1, delta=0.01)

    def test_more_cores_requires_smaller_spacing(self):
        """More cores in same radius should require smaller spacing."""
        spacing_n1 = predict_core_spacing(radius=100, num_cores=1000)
        spacing_n2 = predict_core_spacing(radius=100, num_cores=10000)
        self.assertLess(spacing_n2, spacing_n1)

    def test_circle_smaller_spacing_than_square(self):
        """Circle shape allows smaller spacing than square for same core count."""
        spacing_circle = predict_core_spacing(radius=100, num_cores=10000, shape='circle', packing='hex')
        spacing_square = predict_core_spacing(radius=100, num_cores=10000, shape='square', packing='hex')
        self.assertLess(spacing_circle, spacing_square)

    def test_predicted_spacing_with_lattice_generation(self):
        """Use predict_core_spacing to generate a lattice and verify core count."""
        radius = 50
        target_cores = 5000
        predicted_spacing = predict_core_spacing(radius=radius, num_cores=target_cores, shape='circle', packing='hex')
        coreX, coreY = regular_core_lattice(spacing=predicted_spacing, radius=radius, shape='circle', packing='hex')
        # Allow 10% error since prediction is approximate
        self.assertAlmostEqual(len(coreX), target_cores, delta=target_cores * 0.1)


class TestPredictRadiusAndSpacingInverse(unittest.TestCase):
    """Test that predict_bundle_radius and predict_core_spacing are approximate inverses."""

    def test_radius_spacing_roundtrip_circle_hex(self):
        """Round-trip: radius -> spacing -> radius should return approximately the same."""
        spacing_orig = 2.5
        num_cores = 5000
        radius = predict_bundle_radius(spacing=spacing_orig, num_cores=num_cores, shape='circle', packing='hex')
        spacing_back = predict_core_spacing(radius=radius, num_cores=num_cores, shape='circle', packing='hex')
        self.assertAlmostEqual(spacing_orig, spacing_back, delta=spacing_orig * 0.05)

    def test_radius_spacing_roundtrip_square_hex(self):
        """Round-trip: radius -> spacing -> radius should return approximately the same."""
        spacing_orig = 2.5
        num_cores = 5000
        radius = predict_bundle_radius(spacing=spacing_orig, num_cores=num_cores, shape='square', packing='hex')
        spacing_back = predict_core_spacing(radius=radius, num_cores=num_cores, shape='square', packing='hex')
        self.assertAlmostEqual(spacing_orig, spacing_back, delta=spacing_orig * 0.05)

    def test_spacing_radius_roundtrip_circle_hex(self):
        """Round-trip: spacing -> radius -> spacing should return approximately the same."""
        radius_orig = 50
        num_cores = 5000
        spacing = predict_core_spacing(radius=radius_orig, num_cores=num_cores, shape='circle', packing='hex')
        radius_back = predict_bundle_radius(spacing=spacing, num_cores=num_cores, shape='circle', packing='hex')
        self.assertAlmostEqual(radius_orig, radius_back, delta=radius_orig * 0.05)

    def test_spacing_radius_roundtrip_square_square(self):
        """Round-trip with square/square parameters."""
        radius_orig = 50
        num_cores = 5000
        spacing = predict_core_spacing(radius=radius_orig, num_cores=num_cores, shape='square', packing='square')
        radius_back = predict_bundle_radius(spacing=spacing, num_cores=num_cores, shape='square', packing='square')
        self.assertAlmostEqual(radius_orig, radius_back, delta=radius_orig * 0.05)


if __name__ == '__main__':
    unittest.main()
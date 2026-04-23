# -*- coding: utf-8 -*-
"""
Unit tests for pybundle.bundle_analysis.core_pattern_statistics.
"""

import unittest
import numpy as np

import context

from pybundle.bundle_analysis import core_pattern_statistics
from pybundle.sim import regular_core_lattice


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hex_lattice(spacing=3.0, radius=50.0):
    """Return coreX, coreY for a clean hex lattice clipped to a circle."""
    coreX, coreY = regular_core_lattice(spacing=spacing, radius=radius, packing='hex')
    r = np.sqrt(coreX ** 2 + coreY ** 2)
    mask = r <= radius
    return coreX[mask], coreY[mask]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCorePatternsStatisticsReturnStructure(unittest.TestCase):
    """Return dict must contain all documented keys with correct types/shapes."""

    @classmethod
    def setUpClass(cls):
        coreX, coreY = _make_hex_lattice()
        cls.coreX = coreX
        cls.coreY = coreY
        cls.stats = core_pattern_statistics(coreX, coreY, exclude_boundary=True)

    def test_returns_dict(self):
        self.assertIsInstance(self.stats, dict)

    def test_required_scalar_keys_present(self):
        required = [
            'num_input_cores', 'num_cores', 'num_boundary_cores',
            'mean_nearest_neighbour_dist', 'std_nearest_neighbour_dist',
            'cv_nearest_neighbour_dist', 'mean_local_psi6_mag', 'global_psi6_mag',
            'defect_fraction', 'coord_hist',
        ]
        for key in required:
            with self.subTest(key=key):
                self.assertIn(key, self.stats)

    def test_required_array_keys_present(self):
        for key in ('analysis_mask', 'boundary_mask', 'nearest_neighbour_dists',
                    'local_psi6', 'local_psi6_mag', 'coord_num'):
            with self.subTest(key=key):
                self.assertIn(key, self.stats)

    def test_mask_lengths_match_num_input_cores(self):
        n = self.stats['num_input_cores']
        self.assertEqual(len(self.stats['analysis_mask']), n)
        self.assertEqual(len(self.stats['boundary_mask']), n)

    def test_boundary_and_analysis_masks_are_complementary(self):
        self.assertTrue(np.all(
            self.stats['analysis_mask'] == ~self.stats['boundary_mask']
        ))

    def test_num_cores_matches_analysis_mask(self):
        self.assertEqual(
            self.stats['num_cores'],
            int(np.sum(self.stats['analysis_mask']))
        )

    def test_num_boundary_cores_matches_boundary_mask(self):
        self.assertEqual(
            self.stats['num_boundary_cores'],
            int(np.sum(self.stats['boundary_mask']))
        )

    def test_per_core_array_lengths_match_num_cores(self):
        n = self.stats['num_cores']
        for key in ('nearest_neighbour_dists', 'local_psi6', 'local_psi6_mag', 'coord_num'):
            with self.subTest(key=key):
                self.assertEqual(len(self.stats[key]), n)

    def test_coord_hist_is_dict_of_ints(self):
        ch = self.stats['coord_hist']
        self.assertIsInstance(ch, dict)
        for k, v in ch.items():
            self.assertIsInstance(k, int)
            self.assertIsInstance(v, int)


class TestCorePatternsStatisticsHexLatticeValues(unittest.TestCase):
    """On a near-perfect hex lattice the statistics should hit expected ranges."""

    @classmethod
    def setUpClass(cls):
        cls.spacing = 3.0
        coreX, coreY = _make_hex_lattice(spacing=cls.spacing, radius=60.0)
        cls.stats = core_pattern_statistics(coreX, coreY, exclude_boundary=True)

    def test_mean_nn_dist_close_to_spacing(self):
        self.assertAlmostEqual(
            self.stats['mean_nearest_neighbour_dist'], self.spacing, delta=0.1
        )

    def test_cv_is_low_for_regular_lattice(self):
        """Coefficient of variation should be very small for a regular lattice."""
        self.assertLess(self.stats['cv_nearest_neighbour_dist'], 0.05)

    def test_mean_psi6_mag_close_to_1_for_hex_lattice(self):
        """Interior cores of a hex lattice should have |psi6| close to 1."""
        self.assertGreater(self.stats['mean_local_psi6_mag'], 0.9)

    def test_global_psi6_mag_close_to_1_for_hex_lattice(self):
        self.assertGreater(self.stats['global_psi6_mag'], 0.9)

    def test_defect_fraction_near_zero_for_hex_lattice(self):
        self.assertLess(self.stats['defect_fraction'], 0.1)

    def test_dominant_coordination_number_is_6(self):
        ch = self.stats['coord_hist']
        dominant = max(ch, key=ch.get)
        self.assertEqual(dominant, 6)

    def test_nn_dists_are_positive(self):
        self.assertTrue(np.all(self.stats['nearest_neighbour_dists'] > 0))

    def test_psi6_magnitudes_in_0_1_range(self):
        mags = self.stats['local_psi6_mag']
        self.assertTrue(np.all(mags >= 0.0))
        self.assertTrue(np.all(mags <= 1.0 + 1e-9))


class TestCorePatternsStatisticsBoundaryExclusion(unittest.TestCase):
    """Disabling boundary exclusion should return more cores."""

    def test_exclude_boundary_false_returns_more_cores(self):
        coreX, coreY = _make_hex_lattice()
        stats_excl = core_pattern_statistics(coreX, coreY, exclude_boundary=True)
        stats_full = core_pattern_statistics(coreX, coreY, exclude_boundary=False)
        self.assertGreaterEqual(stats_full['num_cores'], stats_excl['num_cores'])

    def test_no_boundary_cores_when_exclusion_disabled(self):
        coreX, coreY = _make_hex_lattice()
        stats = core_pattern_statistics(coreX, coreY, exclude_boundary=False)
        self.assertEqual(stats['num_boundary_cores'], 0)
        self.assertTrue(np.all(stats['analysis_mask']))

    def test_boundary_layers_increases_exclusion(self):
        coreX, coreY = _make_hex_lattice(radius=60.0)
        stats1 = core_pattern_statistics(coreX, coreY, boundary_layers=1)
        stats2 = core_pattern_statistics(coreX, coreY, boundary_layers=2)
        self.assertGreater(stats2['num_boundary_cores'], stats1['num_boundary_cores'])


class TestCorePatternsStatisticsInputValidation(unittest.TestCase):
    """Invalid inputs must raise ValueError."""

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(ValueError):
            core_pattern_statistics(np.arange(10), np.arange(9))

    def test_too_few_cores_raises(self):
        with self.assertRaises(ValueError):
            core_pattern_statistics([0, 1, 2], [0, 1, 2])

    def test_invalid_k_neighbours_raises(self):
        coreX, coreY = _make_hex_lattice()
        with self.assertRaises(ValueError):
            core_pattern_statistics(coreX, coreY, k_neighbours=0)

    def test_invalid_boundary_layers_raises(self):
        coreX, coreY = _make_hex_lattice()
        with self.assertRaises(ValueError):
            core_pattern_statistics(coreX, coreY, boundary_layers=0)


class TestCorePatternsStatisticsNoisyLattice(unittest.TestCase):
    """Adding positional noise degrades order and increases CV."""

    @classmethod
    def setUpClass(cls):
        spacing = 3.0
        rng = np.random.default_rng(7)
        coreX, coreY = _make_hex_lattice(spacing=spacing, radius=60.0)
        noise = rng.normal(0.0, 0.4 * spacing, size=(2, len(coreX)))
        cls.stats_clean = core_pattern_statistics(coreX, coreY, exclude_boundary=True)
        cls.stats_noisy = core_pattern_statistics(
            coreX + noise[0], coreY + noise[1], exclude_boundary=True
        )

    def test_noise_increases_cv(self):
        self.assertGreater(
            self.stats_noisy['cv_nearest_neighbour_dist'],
            self.stats_clean['cv_nearest_neighbour_dist']
        )

    def test_noise_reduces_psi6(self):
        self.assertLess(
            self.stats_noisy['mean_local_psi6_mag'],
            self.stats_clean['mean_local_psi6_mag']
        )


if __name__ == '__main__':
    unittest.main()

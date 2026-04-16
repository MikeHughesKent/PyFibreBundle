# -*- coding: utf-8 -*-
"""
Unit tests for pybundle.core_interpolation module.

Tests cover:
    - core_values
    - find_cores (using synthetically generated bundle images)
    - calib_tri_interp (using synthetically generated bundle images)
    - recon_tri_interp (using synthetically generated bundle images)

A synthetic bundle image is generated using regularly-spaced Gaussian spots
to avoid depending on external test data files.

"""

import unittest
import math
import numpy as np

import context


from pybundle.core_interpolation import (
    core_values,
    find_cores,
    calib_tri_interp,
    recon_tri_interp,
)
from pybundle.bundle_calibration import BundleCalibration

from pybundle import extract_central

import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bundle_image(img_size=200, spacing=10, core_sigma=1.5):
    """Return a synthetic fibre-bundle image (float32) and the true core positions.

    Cores are placed on a hexagonal grid inside a circular region.  Each core
    is rendered as a 2-D Gaussian blob.

    Parameters
    ----------
    img_size : int
        Side length of the returned square image.
    spacing : float
        Centre-to-centre core spacing in pixels.
    core_sigma : float
        Sigma of the Gaussian used to render each core.

    Returns
    -------
    img : ndarray, shape (img_size, img_size), dtype float32
    trueX : ndarray  – true x positions of all cores
    trueY : ndarray  – true y positions of all cores
    """
    radius = img_size // 2 - spacing
    cx = cy = img_size // 2

    # Hex-grid core positions centred on (cx, cy)
    numX = int(radius / spacing) * 2 + 1
    numY = round(numX * 2 / math.sqrt(3))
    gX, gY = np.meshgrid(
        np.linspace(-radius, radius, numX),
        np.linspace(-radius, radius, numY),
    )
    gX[1::2] += spacing / 2.0
    trueX = gX.ravel() + cx
    trueY = gY.ravel() + cy

    # Keep only cores inside the bundle circle
    inside = (trueX - cx) ** 2 + (trueY - cy) ** 2 < radius ** 2
    trueX = trueX[inside]
    trueY = trueY[inside]

    # Render Gaussian spots
    img = np.zeros((img_size, img_size), dtype=np.float32)
    half = int(math.ceil(core_sigma * 4))
    for px, py in zip(trueX, trueY):
        ipx, ipy = int(round(px)), int(round(py))
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                xi, yi = ipx + dx, ipy + dy
                if 0 <= xi < img_size and 0 <= yi < img_size:
                    img[yi, xi] += math.exp(-(dx ** 2 + dy ** 2) / (2 * core_sigma ** 2))

    return img, trueX, trueY


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCoreValues(unittest.TestCase):

    def test_known_values_extracted_correctly(self):
        """core_values should return pixel values at specified integer positions."""
        img = np.zeros((50, 50), dtype=np.float32)
        img[10, 15] = 7.0
        img[25, 30] = 13.0
        coreX = np.array([15, 30], dtype=np.uint16)
        coreY = np.array([10, 25], dtype=np.uint16)
        vals = core_values(img, coreX, coreY, filterSize=None)
        self.assertAlmostEqual(float(vals[0]), 7.0, places=4)
        self.assertAlmostEqual(float(vals[1]), 13.0, places=4)

    def test_output_length_matches_number_of_cores(self):
        img = np.random.rand(60, 60).astype(np.float32)
        coreX = np.array([10, 20, 30, 40], dtype=np.uint16)
        coreY = np.array([10, 20, 30, 40], dtype=np.uint16)
        vals = core_values(img, coreX, coreY, filterSize=None)
        self.assertEqual(len(vals), 4)

    def test_gaussian_filter_option_does_not_change_shape(self):
        """Passing a filterSize should still return one value per core."""
        img = np.random.rand(100, 100).astype(np.float32)
        coreX = np.array([20, 50, 80], dtype=np.uint16)
        coreY = np.array([20, 50, 80], dtype=np.uint16)
        vals = core_values(img, coreX, coreY, filterSize=2.0)
        self.assertEqual(len(vals), 3)

    def test_uniform_image_returns_uniform_values(self):
        """For a uniform image every core should return the same value."""
        img = np.ones((80, 80), dtype=np.float32) * 5.0
        coreX = np.array([10, 30, 50, 70], dtype=np.uint16)
        coreY = np.array([10, 30, 50, 70], dtype=np.uint16)
        vals = core_values(img, coreX, coreY, filterSize=None)
        np.testing.assert_allclose(vals, 5.0, rtol=1e-5)


class TestFindCores(unittest.TestCase):
    """find_cores tests use a synthetically generated bundle image."""

    @classmethod
    def setUpClass(cls):
        cls.spacing = 12
        cls.img_size = 200
        cls.img, cls.trueX, cls.trueY = _make_bundle_image(
            img_size=cls.img_size, spacing=cls.spacing
        )

    def test_returns_two_arrays(self):
        cx, cy = find_cores(self.img, self.spacing)
        self.assertIsInstance(cx, np.ndarray)
        self.assertIsInstance(cy, np.ndarray)

    def test_arrays_same_length(self):
        cx, cy = find_cores(self.img, self.spacing)
        self.assertEqual(len(cx), len(cy))

    def test_found_core_count_approximately_correct(self):
        """The number of found cores should be close to the number generated."""
        cx, cy = find_cores(self.img, self.spacing)
        expected = len(self.trueX)
        self.assertAlmostEqual(len(cx), expected, delta=max(5, expected // 10))

    def test_cores_within_image_bounds(self):
        """All found core positions should be within the image dimensions."""
        cx, cy = find_cores(self.img, self.spacing)
        self.assertTrue(np.all(cx >= 0))
        self.assertTrue(np.all(cy >= 0))
        self.assertTrue(np.all(cx < self.img_size))
        self.assertTrue(np.all(cy < self.img_size))


class TestCalibTriInterp(unittest.TestCase):
    """calib_tri_interp tests use a synthetically generated bundle image."""

    @classmethod
    def setUpClass(cls):
        spacing = 10
        cls.img_size = 180
        cls.grid_size = 64
        cls.img, _, _ = _make_bundle_image(img_size=cls.img_size, spacing=spacing)
        cx = cy = cls.img_size // 2
        radius = cls.img_size // 2 - spacing
        # Normalise to 0-255 range so auto-masking thresholding works
        cls.img = (cls.img / cls.img.max() * 255).astype(np.float32)
        cls.calib = calib_tri_interp(
            cls.img,
            coreSize=spacing,
            gridSize=cls.grid_size,
            centreX=cx,
            centreY=cy,
            radius=radius,
            autoMask=False,
        )

    def test_returns_bundle_calibration(self):
        self.assertIsInstance(self.calib, BundleCalibration)

    def test_calibration_has_core_positions(self):
        self.assertIsNotNone(self.calib.coreX)
        self.assertIsNotNone(self.calib.coreY)

    def test_core_position_arrays_same_length(self):
        self.assertEqual(len(self.calib.coreX), len(self.calib.coreY))

    def test_grid_size_stored(self):
        self.assertEqual(self.calib.gridSize, self.grid_size)

    def test_radius_stored(self):
        expected = self.img_size // 2 - 10
        self.assertAlmostEqual(self.calib.radius, expected, delta=1)

    def test_triangulation_present(self):
        self.assertIsNotNone(self.calib.tri)

    def test_barycentric_coords_present(self):
        self.assertIsNotNone(self.calib.baryCoords)

    def test_non_colour_flag(self):
        self.assertFalse(self.calib.col)


class TestReconTriInterp(unittest.TestCase):
    """recon_tri_interp tests use a synthetically generated bundle image and calibration."""

    @classmethod
    def setUpClass(cls):
        spacing = 10
        img_size = 180
        cls.grid_size = 64
        img, _, _ = _make_bundle_image(img_size=img_size, spacing=spacing)
        img = (img / img.max() * 255).astype(np.float32)
        cx = cy = img_size // 2
        radius = img_size // 2 - spacing
        cls.calib = calib_tri_interp(
            img,
            coreSize=spacing,
            gridSize=cls.grid_size,
            centreX=cx,
            centreY=cy,
            radius=radius,
            autoMask=False,
        )
        cls.img = img

    def test_output_shape_matches_grid_size(self):
        """Reconstructed image should be (gridSize, gridSize)."""
        out = recon_tri_interp(self.img, self.calib, numba=False)
        self.assertEqual(out.shape, (self.grid_size, self.grid_size))

    def test_output_is_2d_for_greyscale(self):
        out = recon_tri_interp(self.img, self.calib, numba=False)
        self.assertEqual(out.ndim, 2)

    def test_output_contains_nonzero_values(self):
        """Reconstructed image should not be entirely zero."""
        out = recon_tri_interp(self.img, self.calib, numba=False)
        self.assertGreater(float(np.max(out)), 0.0)

    def test_bright_uniform_image_produces_positive_output(self):
        """A uniformly bright input image should yield a uniformly positive reconstruction."""
        uniform_img = np.ones_like(self.img) * 200.0
        out = recon_tri_interp(uniform_img, self.calib, numba=False)
        # Inside the mask, values should be positive
        plt.imshow(out)
        if self.calib.mask is not None:
            inside = self.calib.mask.astype(bool)
            self.assertTrue(np.all(extract_central(out, 20) > 0))
        else:
            self.assertGreater(float(np.mean(out)), 0.0)

    def test_darker_image_gives_smaller_output(self):
        """Halving the image brightness should approximately halve the output."""
        out_full = recon_tri_interp(self.img, self.calib, numba=False)
        out_half = recon_tri_interp(self.img * 0.5, self.calib, numba=False)
        ratio = float(np.mean(out_full[out_full > 0])) / float(np.mean(out_half[out_half > 0]))
        self.assertAlmostEqual(ratio, 2.0, delta=0.3)


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""
Unit tests for the PyBundle class.

Each test verifies that PyBundle.process() produces output that is
numerically equal (or very close) to calling the equivalent sequence of
low-level pybundle functions directly.

Tests cover:
    - FILTER method (Gaussian filter, mask, crop)
    - FILTER method with normalisation image
    - FILTER method with background subtraction
    - EDGE_FILTER method
    - TRILIN method (triangular linear interpolation)
    - Output type casting (uint8, uint16)
    - autoContrast flag

A synthetic fibre-bundle image is generated using regularly-spaced Gaussian
spots so that no external test-data files are needed.

"""


import unittest
import math
import numpy as np

import context



from pybundle import PyBundle
from pybundle.core import (
    g_filter,
    normalise_image,
    apply_mask,
    crop_rect,
    get_mask,
    find_bundle,
    edge_filter,
    filter_image,
)
from pybundle.core_interpolation import recon_tri_interp


# ---------------------------------------------------------------------------
# Synthetic bundle image helper (reused from test_core_interpolation.py style)
# ---------------------------------------------------------------------------

def _make_bundle_image(img_size=200, spacing=10, core_sigma=1.5, intensity=200.0):
    """Return a synthetic fibre-bundle image (float32) with Gaussian cores on a
    hex grid inside a circular region.

    Returns
    -------
    img : ndarray, shape (img_size, img_size), dtype float32
    """
    radius = img_size // 2 - spacing
    cx = cy = img_size // 2

    numX = int(radius / spacing) * 2 + 1
    numY = round(numX * 2 / math.sqrt(3))
    gX, gY = np.meshgrid(
        np.linspace(-radius, radius, numX),
        np.linspace(-radius, radius, numY),
    )
    gX[1::2] += spacing / 2.0
    trueX = gX.ravel() + cx
    trueY = gY.ravel() + cy

    inside = (trueX - cx) ** 2 + (trueY - cy) ** 2 < radius ** 2
    trueX = trueX[inside]
    trueY = trueY[inside]

    img = np.zeros((img_size, img_size), dtype=np.float32)
    half = int(math.ceil(core_sigma * 4))
    for px, py in zip(trueX, trueY):
        ipx, ipy = int(round(px)), int(round(py))
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                xi, yi = ipx + dx, ipy + dy
                if 0 <= xi < img_size and 0 <= yi < img_size:
                    img[yi, xi] += intensity * math.exp(
                        -(dx ** 2 + dy ** 2) / (2 * core_sigma ** 2)
                    )
    return img


# ---------------------------------------------------------------------------
# Shared fixtures (computed once per class via setUpClass)
# ---------------------------------------------------------------------------

IMG_SIZE = 180
SPACING = 10


class _BundleFixture(unittest.TestCase):
    """Mixin that creates shared synthetic images in setUpClass."""

    @classmethod
    def setUpClass(cls):
        cls.img = _make_bundle_image(img_size=IMG_SIZE, spacing=SPACING)
        cls.calib_img = _make_bundle_image(
            img_size=IMG_SIZE, spacing=SPACING, intensity=255.0
        )
        cls.loc = find_bundle(cls.calib_img)
        cls.mask = get_mask(cls.calib_img, cls.loc)


# ---------------------------------------------------------------------------
# FILTER method tests
# ---------------------------------------------------------------------------

class TestPyBundleFilter(_BundleFixture):
    """PyBundle FILTER method should match sequential low-level calls."""

    def _low_level_filter(self, img, filter_size, loc, mask, crop=True):
        """Replicate what __process_filter does."""
        out = g_filter(img, filter_size)
        out = apply_mask(out, mask)
        if crop:
            out = crop_rect(out, loc)[0]
        return out

    def test_filter_with_crop_and_mask_matches_low_level(self):
        filter_size = 2.5
        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=filter_size,
            crop=True,
            applyMask=True,
            autoLoc=False,
            autoMask=False,
        )
        pyb.set_loc(self.loc)
        pyb.set_mask(self.mask)

        result_oop = pyb.process(self.img)
        result_ll = self._low_level_filter(
            self.img, filter_size, self.loc, self.mask, crop=True
        )

        np.testing.assert_allclose(
            result_oop.astype(np.float64),
            result_ll.astype(np.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_filter_no_crop_no_mask_matches_low_level(self):
        filter_size = 2.0
        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=filter_size,
            crop=False,
            applyMask=False,
            autoLoc=False,
            autoMask=False,
        )
        pyb.set_loc(self.loc)

        result_oop = pyb.process(self.img)
        result_ll = g_filter(self.img, filter_size)

        np.testing.assert_allclose(
            result_oop.astype(np.float64),
            result_ll.astype(np.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_filter_with_mask_only_matches_low_level(self):
        filter_size = 3.0
        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=filter_size,
            crop=False,
            applyMask=True,
            autoLoc=False,
            autoMask=False,
        )
        pyb.set_loc(self.loc)
        pyb.set_mask(self.mask)

        result_oop = pyb.process(self.img)
        result_ll = self._low_level_filter(
            self.img, filter_size, self.loc, self.mask, crop=False
        )

        np.testing.assert_allclose(
            result_oop.astype(np.float64),
            result_ll.astype(np.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_filter_output_shape_matches_low_level(self):
        filter_size = 2.5
        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=filter_size,
            crop=True,
            applyMask=True,
            autoLoc=False,
            autoMask=False,
        )
        pyb.set_loc(self.loc)
        pyb.set_mask(self.mask)

        result_oop = pyb.process(self.img)
        result_ll = self._low_level_filter(
            self.img, filter_size, self.loc, self.mask, crop=True
        )

        self.assertEqual(result_oop.shape, result_ll.shape)


class TestPyBundleFilterNormalise(_BundleFixture):
    """PyBundle FILTER with normalisation should match low-level normalise_image."""

    def test_filter_with_normalisation_matches_low_level(self):
        filter_size = 2.5
        
        img = np.random.rand(20,20)
        norm_img = np.random.rand(20,20)

        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=filter_size,
            crop=False,
            applyMask=False,
            autoLoc=False,
            autoMask=False,
            normaliseImage=norm_img,
        )

        result_oop = pyb.process(img)

        # Low-level equivalent: filter both images, then normalise
        img_f = g_filter(img, filter_size)
        norm_f = g_filter(norm_img, filter_size)
        result_ll = normalise_image(img_f, norm_f)
     
        np.testing.assert_allclose(
            result_oop.astype(np.float64),
            result_ll.astype(np.float64),
            rtol=1e-4,
            atol=1e-4,
        )


class TestPyBundleFilterBackground(_BundleFixture):
    """PyBundle FILTER with background subtraction should match low-level subtraction."""

    def test_filter_with_background_matches_low_level(self):
        filter_size = 2.5
        background = self.calib_img * 0.1  # simulate a dim background

        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=filter_size,
            crop=False,
            applyMask=False,
            autoLoc=False,
            autoMask=False,
        )
        pyb.set_loc(self.loc)
        pyb.set_background(background)

        result_oop = pyb.process(self.img)

        # Low-level: set_background casts to float64 internally, match that here
        bg_f64 = background.astype('float')
        img_sub = self.img.astype('float') - bg_f64
        result_ll = g_filter(img_sub, filter_size)

        np.testing.assert_allclose(
            result_oop.astype(np.float64),
            result_ll.astype(np.float64),
            rtol=1e-4,
            atol=1e-4,
        )


# ---------------------------------------------------------------------------
# EDGE_FILTER method tests
# ---------------------------------------------------------------------------

class TestPyBundleEdgeFilter(_BundleFixture):
    """PyBundle EDGE_FILTER method should match manual crop + edge_filter + filter_image."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Pre-compute edge filter parameters based on bundle radius
        cls.filter_size = cls.loc[2]
        cls.edge_pos = 4.0
        cls.edge_slope = 0.4

    def _low_level_edge_filter(self, img, loc, edge_pos, edge_slope):
        """Replicate __process_edge_filter: mask -> crop -> build filter -> apply."""
        out = apply_mask(img, self.mask)
        out, new_loc = crop_rect(out, loc)
        filt_size = new_loc[2] * 2
        filt = edge_filter(filt_size, edge_pos, edge_slope)
        out = filter_image(out, filt)
        return out

    def test_edge_filter_matches_low_level(self):
        pyb = PyBundle(
            coreMethod=PyBundle.EDGE_FILTER,
            crop=True,
            applyMask=True,
            autoLoc=False,
            autoMask=False,
        )
        pyb.set_loc(self.loc)
        pyb.set_mask(self.mask)
        pyb.set_edge_filter_shape(self.edge_pos, self.edge_slope)

        result_oop = pyb.process(self.img)
        result_ll = self._low_level_edge_filter(
            self.img, self.loc, self.edge_pos, self.edge_slope
        )

        np.testing.assert_allclose(
            result_oop.astype(np.float64),
            result_ll.astype(np.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_edge_filter_output_shape_is_cropped(self):
        """Output of edge filter should have same shape as the cropped region."""
        pyb = PyBundle(
            coreMethod=PyBundle.EDGE_FILTER,
            crop=True,
            applyMask=True,
            autoLoc=False,
            autoMask=False,
        )
        pyb.set_loc(self.loc)
        pyb.set_mask(self.mask)
        pyb.set_edge_filter_shape(self.edge_pos, self.edge_slope)

        result_oop = pyb.process(self.img)
        expected_size = self.loc[2] * 2
        self.assertAlmostEqual(result_oop.shape[0], expected_size, delta=2)
        self.assertAlmostEqual(result_oop.shape[1], expected_size, delta=2)


# ---------------------------------------------------------------------------
# TRILIN method tests
# ---------------------------------------------------------------------------

class TestPyBundleTrilin(_BundleFixture):
    """PyBundle TRILIN method should match calib_tri_interp + recon_tri_interp."""

    GRID_SIZE = 64

    def test_trilin_output_shape_matches_grid_size(self):
        pyb = PyBundle(
            coreMethod=PyBundle.TRILIN,
            gridSize=self.GRID_SIZE,
            useNumba=False,
        )
        pyb.set_calib_image(self.calib_img)
        pyb.calibrate()
        result = pyb.process(self.img)
        self.assertEqual(result.shape, (self.GRID_SIZE, self.GRID_SIZE))

    def test_trilin_matches_low_level_recon(self):
        """PyBundle TRILIN result must match directly calling recon_tri_interp."""
        pyb = PyBundle(
            coreMethod=PyBundle.TRILIN,
            gridSize=self.GRID_SIZE,
            useNumba=False,
        )
        pyb.set_calib_image(self.calib_img)
        pyb.calibrate()

        result_oop = pyb.process(self.img)
        result_ll = recon_tri_interp(self.img, pyb.calibration, numba=False)

        np.testing.assert_allclose(
            result_oop.astype(np.float64),
            result_ll.astype(np.float64),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_trilin_produces_nonzero_output(self):
        pyb = PyBundle(
            coreMethod=PyBundle.TRILIN,
            gridSize=self.GRID_SIZE,
            useNumba=False,
        )
        pyb.set_calib_image(self.calib_img)
        pyb.calibrate()
        result = pyb.process(self.img)
        self.assertGreater(float(np.max(result)), 0.0)

    def test_trilin_calibration_stored_after_calibrate(self):
        pyb = PyBundle(
            coreMethod=PyBundle.TRILIN,
            gridSize=self.GRID_SIZE,
            useNumba=False,
        )
        pyb.set_calib_image(self.calib_img)
        pyb.calibrate()
        self.assertIsNotNone(pyb.calibration)


# ---------------------------------------------------------------------------
# Output type and autoContrast tests
# ---------------------------------------------------------------------------

class TestPyBundleOutputTypes(_BundleFixture):
    """PyBundle output type casting should produce correctly-typed arrays."""

    def _make_filter_pyb(self, output_type, auto_contrast=False):
        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=2.5,
            crop=False,
            applyMask=False,
            autoLoc=False,
            autoMask=False,
            outputType=output_type,
            autoContrast=auto_contrast,
        )
        pyb.set_loc(self.loc)
        return pyb

    def test_uint8_output_dtype(self):
        pyb = self._make_filter_pyb('uint8', auto_contrast=True)
        result = pyb.process(self.img)
        self.assertEqual(result.dtype, np.uint8)

    def test_uint16_output_dtype(self):
        pyb = self._make_filter_pyb('uint16', auto_contrast=True)
        result = pyb.process(self.img)
        self.assertEqual(result.dtype, np.uint16)

    def test_float64_output_dtype(self):
        pyb = self._make_filter_pyb('float64')
        result = pyb.process(self.img)
        self.assertEqual(result.dtype, np.float64)

    def test_uint8_range_0_to_255(self):
        pyb = self._make_filter_pyb('uint8', auto_contrast=True)
        result = pyb.process(self.img)
        self.assertGreaterEqual(int(np.min(result)), 0)
        self.assertLessEqual(int(np.max(result)), 255)

    def test_uint16_range_0_to_65535(self):
        pyb = self._make_filter_pyb('uint16', auto_contrast=True)
        result = pyb.process(self.img)
        self.assertGreaterEqual(int(np.min(result)), 0)
        self.assertLessEqual(int(np.max(result)), 65535)


class TestPyBundleAutoContrast(_BundleFixture):
    """autoContrast should stretch output to use the full output-type range."""

    def test_autocontrast_uint8_min_is_zero(self):
        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=2.5,
            crop=False,
            applyMask=False,
            autoLoc=False,
            autoMask=False,
            outputType='uint8',
            autoContrast=True,
        )
        pyb.set_loc(self.loc)
        result = pyb.process(self.img)
        self.assertEqual(int(np.min(result)), 0)

    def test_autocontrast_uint8_max_is_255(self):
        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=2.5,
            crop=False,
            applyMask=False,
            autoLoc=False,
            autoMask=False,
            outputType='uint8',
            autoContrast=True,
        )
        pyb.set_loc(self.loc)
        result = pyb.process(self.img)
        self.assertEqual(int(np.max(result)), 255)

    def test_no_autocontrast_does_not_force_255_max(self):
        """Without autoContrast, the maximum is not necessarily 255."""
        pyb = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=2.5,
            crop=False,
            applyMask=False,
            autoLoc=False,
            autoMask=False,
            outputType='uint8',
            autoContrast=False,
        )
        pyb.set_loc(self.loc)
        # The image values are float intensities that may exceed 255 before cast
        result_ac_off = pyb.process(self.img)

        pyb2 = PyBundle(
            coreMethod=PyBundle.FILTER,
            filterSize=2.5,
            crop=False,
            applyMask=False,
            autoLoc=False,
            autoMask=False,
            outputType='uint8',
            autoContrast=True,
        )
        pyb2.set_loc(self.loc)
        result_ac_on = pyb2.process(self.img)

        # They should generally differ (autoContrast rescales)
        self.assertFalse(np.array_equal(result_ac_off, result_ac_on))


# ---------------------------------------------------------------------------
# Calibrate method
# ---------------------------------------------------------------------------

class TestPyBundleCalibrate(_BundleFixture):
    """calibrate() should populate loc, mask and, for TRILIN, calibration."""

    def test_calibrate_filter_sets_loc(self):
        pyb = PyBundle(coreMethod=PyBundle.FILTER, filterSize=2.5)
        pyb.set_calib_image(self.calib_img)
        pyb.calibrate()
        self.assertIsNotNone(pyb.loc)

    def test_calibrate_filter_sets_mask(self):
        pyb = PyBundle(coreMethod=PyBundle.FILTER, filterSize=2.5)
        pyb.set_calib_image(self.calib_img)
        pyb.calibrate()
        self.assertIsNotNone(pyb.mask)

    def test_calibrate_trilin_sets_calibration(self):
        pyb = PyBundle(
            coreMethod=PyBundle.TRILIN,
            gridSize=64,
            useNumba=False,
        )
        pyb.set_calib_image(self.calib_img)
        pyb.calibrate()
        self.assertIsNotNone(pyb.calibration)

    def test_calibrate_edge_filter_sets_edge_filter(self):
        pyb = PyBundle(coreMethod=PyBundle.EDGE_FILTER)
        pyb.set_calib_image(self.calib_img)
        pyb.set_edge_filter_shape(4.0, 0.4)
        pyb.calibrate()
        self.assertIsNotNone(pyb.edgeFilter)


if __name__ == '__main__':
    unittest.main()

# -*- coding: utf-8 -*-
"""
Unit tests for pybundle.core module.

Tests cover:
    - g_filter
    - median_filter
    - normalise_image
    - edge_filter
    - filter_image
    - get_mask
    - apply_mask
    - crop_rect

"""

import unittest
import numpy as np


from pybundle.core import (
    g_filter,
    median_filter,
    normalise_image,
    edge_filter,
    filter_image,
    get_mask,
    apply_mask,
    crop_rect,
)


class TestGFilter(unittest.TestCase):

    def test_output_shape_2d(self):
        """Output shape matches 2D input shape."""
        img = np.random.rand(100, 100).astype(np.float32)
        out = g_filter(img, filterSize=3)
        self.assertEqual(out.shape, img.shape)

    def test_output_shape_3d(self):
        """Output shape matches 3D colour input shape."""
        img = np.random.rand(100, 100, 3).astype(np.float32)
        out = g_filter(img, filterSize=3)
        self.assertEqual(out.shape, img.shape)

    def test_filter_reduces_variance(self):
        """Filtering a noisy image should reduce variance."""
        rng = np.random.default_rng(42)
        img = rng.random((100, 100)).astype(np.float32)
        out = g_filter(img, filterSize=5)
        self.assertLess(float(np.var(out)), float(np.var(img)))

    def test_uniform_image_unchanged(self):
        """Gaussian filter leaves a uniform image unchanged."""
        img = np.ones((50, 50), dtype=np.float32) * 42.0
        out = g_filter(img, filterSize=3)
        np.testing.assert_allclose(out, 42.0, rtol=1e-4)

    def test_custom_kernel_size(self):
        """Explicit kernelSize parameter does not raise and preserves shape."""
        img = np.random.rand(60, 60).astype(np.float32)
        out = g_filter(img, filterSize=3, kernelSize=11)
        self.assertEqual(out.shape, img.shape)


class TestMedianFilter(unittest.TestCase):

    def test_output_shape_2d(self):
        img = (np.random.rand(100, 100) * 255).astype(np.uint8)
        out = median_filter(img, filterSize=3)
        self.assertEqual(out.shape, img.shape)

    def test_removes_isolated_spike(self):
        """Median filter should suppress an isolated hot pixel."""
        img = np.zeros((20, 20), dtype=np.uint8)
        img[10, 10] = 255  # single hot pixel
        out = median_filter(img, filterSize=3)
        self.assertEqual(int(out[10, 10]), 0)

    def test_uniform_image_unchanged(self):
        img = np.ones((30, 30), dtype=np.uint8) * 100
        out = median_filter(img, filterSize=3)
        np.testing.assert_array_equal(out, img)


class TestNormaliseImage(unittest.TestCase):

    def test_divide_by_itself_gives_ones(self):
        """Dividing an image by itself should yield 1.0 everywhere (non-zero pixels)."""
        img = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        out = normalise_image(img, img)
        np.testing.assert_allclose(out, 1.0, rtol=1e-5)

    def test_output_dtype_is_float32(self):
        """Without outputType, the result is float32."""
        img = np.ones((10, 10), dtype=np.uint8) * 100
        normImg = np.ones((10, 10), dtype=np.uint8) * 50
        out = normalise_image(img, normImg)
        self.assertEqual(out.dtype, np.float32)

    #def test_zero_pixels_stay_zero(self):
    #    """Pixels where normImg is 0 should remain 0 in the output."""
    #    img = np.ones((4, 4), dtype=np.float32) * 10
    #    normImg = np.zeros((4, 4), dtype=np.float32)
    #    out = normalise_image(img, normImg)
    #    np.testing.assert_array_equal(out, 0)

    def test_scale_factor_applied(self):
        """Dividing by half the value should double each pixel."""
        img = np.array([[100.0, 200.0]], dtype=np.float32)
        normImg = np.array([[50.0, 100.0]], dtype=np.float32)
        out = normalise_image(img, normImg)
        np.testing.assert_allclose(out, [[2.0, 2.0]], rtol=1e-5)

    def test_2d_norm_applied_to_3d_image(self):
        """A 2D normImg should apply to each plane of a 3D image."""
        img = np.ones((10, 10, 3), dtype=np.float32) * 60
        normImg = np.ones((10, 10), dtype=np.float32) * 30
        out = normalise_image(img, normImg)
        np.testing.assert_allclose(out, 2.0, rtol=1e-5)

    def test_output_shape_preserved_2d(self):
        img = np.ones((20, 30), dtype=np.float32)
        normImg = np.ones((20, 30), dtype=np.float32)
        out = normalise_image(img, normImg)
        self.assertEqual(out.shape, (20, 30))

    def test_output_shape_preserved_3d(self):
        img = np.ones((20, 30, 3), dtype=np.float32)
        normImg = np.ones((20, 30), dtype=np.float32)
        out = normalise_image(img, normImg)
        self.assertEqual(out.shape, (20, 30, 3))


class TestEdgeFilter(unittest.TestCase):

    def test_output_shape(self):
        """Output is square with dimensions (imgSize, imgSize)."""
        filt = edge_filter(imgSize=128, edgePos=4, skinThickness=0.1)
        self.assertEqual(filt.shape, (128, 128))

    def test_values_in_range(self):
        """All filter values are between 0 and 1."""
        filt = edge_filter(imgSize=64, edgePos=4, skinThickness=0.1)
        self.assertGreaterEqual(float(np.min(filt)), 0.0)
        self.assertLessEqual(float(np.max(filt)), 1.0)

    def test_center_passes(self):
        """Centre of the filter (DC component) should be 1 (low-pass)."""
        imgSize = 128
        filt = edge_filter(imgSize=imgSize, edgePos=4, skinThickness=0.5)
        centre = filt[imgSize // 2, imgSize // 2]
        self.assertAlmostEqual(float(centre), 1.0, places=5)

    def test_corner_attenuated(self):
        """Corners of the filter (high frequencies) should be 0 (high-pass roll-off)."""
        imgSize = 128
        filt = edge_filter(imgSize=imgSize, edgePos=4, skinThickness=0.1)
        self.assertAlmostEqual(float(filt[0, 0]), 0.0, places=5)


class TestFilterImage(unittest.TestCase):

    def test_output_shape_2d(self):
        """Output shape matches 2D input."""
        imgSize = 64
        img = np.random.rand(imgSize, imgSize).astype(np.float32)
        filt = np.ones((imgSize, imgSize))
        out = filter_image(img, filt)
        self.assertEqual(out.shape, img.shape)

    def test_output_shape_3d(self):
        """Output shape matches 3D colour input."""
        imgSize = 64
        img = np.random.rand(imgSize, imgSize, 3).astype(np.float32)
        filt = np.ones((imgSize, imgSize))
        out = filter_image(img, filt)
        self.assertEqual(out.shape, img.shape)

    def test_all_ones_filter_preserves_image(self):
        """Applying an all-ones frequency-domain filter should approximately preserve the image."""
        imgSize = 64
        img = np.random.rand(imgSize, imgSize).astype(np.float64)
        filt = np.ones((imgSize, imgSize))
        out = filter_image(img, filt)
        np.testing.assert_allclose(out, img, rtol=1e-5, atol=1e-5)

    def test_zero_filter_produces_zero(self):
        """Applying a zero filter should produce zeroed output."""
        imgSize = 64
        img = np.random.rand(imgSize, imgSize)
        filt = np.zeros((imgSize, imgSize))
        out = filter_image(img, filt)
        np.testing.assert_allclose(out, 0.0, atol=1e-10)


class TestGetMask(unittest.TestCase):

    def test_output_shape_matches_image(self):
        img = np.zeros((100, 100))
        loc = (50, 50, 40)
        mask = get_mask(img, loc)
        self.assertEqual(mask.shape, img.shape)

    def test_centre_pixel_is_inside(self):
        """The centre pixel should be inside the mask (value 1)."""
        img = np.zeros((100, 100))
        loc = (50, 50, 40)
        mask = get_mask(img, loc)
        self.assertEqual(int(mask[50, 50]), 1)

    def test_corner_pixel_is_outside(self):
        """The corner pixel should be outside a circle that does not reach it."""
        img = np.zeros((100, 100))
        loc = (50, 50, 30)  # radius 30, corner is ~70 px away
        mask = get_mask(img, loc)
        self.assertEqual(int(mask[0, 0]), 0)

    def test_mask_is_binary(self):
        """Mask should only contain 0 and 1."""
        img = np.zeros((100, 100))
        loc = (50, 50, 40)
        mask = get_mask(img, loc)
        unique_vals = np.unique(mask)
        for v in unique_vals:
            self.assertIn(v, [0, 1])

    def test_pixels_within_radius_inside(self):
        """All pixels within the specified radius should be inside the mask."""
        img = np.zeros((100, 100))
        cx, cy, rad = 50, 50, 20
        loc = (cx, cy, rad)
        mask = get_mask(img, loc)
        # Check a pixel clearly inside the circle
        self.assertEqual(int(mask[cy, cx + rad - 2]), 1)


class TestApplyMask(unittest.TestCase):

    def test_masked_pixels_are_zero(self):
        """Pixels outside the mask should become 0."""
        img = np.ones((100, 100)) * 5.0
        mask = np.zeros((100, 100))
        mask[40:60, 40:60] = 1
        out = apply_mask(img, mask)
        self.assertEqual(float(out[0, 0]), 0.0)

    def test_unmasked_pixels_preserved(self):
        """Pixels inside the mask should keep their original values."""
        img = np.ones((100, 100)) * 7.0
        mask = np.zeros((100, 100))
        mask[40:60, 40:60] = 1
        out = apply_mask(img, mask)
        self.assertAlmostEqual(float(out[50, 50]), 7.0)

    def test_none_mask_returns_image_unchanged(self):
        """A None mask should return the image unchanged."""
        img = np.ones((50, 50)) * 3.0
        out = apply_mask(img, None)
        np.testing.assert_array_equal(out, img)

    def test_3d_image_mask_applied_per_channel(self):
        """Mask should be applied to each channel of a colour image."""
        img = np.ones((50, 50, 3)) * 9.0
        mask = np.zeros((50, 50))
        mask[20:30, 20:30] = 1
        out = apply_mask(img, mask)
        # Outside region should be 0 in all channels
        self.assertEqual(float(out[0, 0, 0]), 0.0)
        self.assertEqual(float(out[0, 0, 1]), 0.0)
        # Inside region should be 9 in all channels
        self.assertAlmostEqual(float(out[25, 25, 2]), 9.0)

    def test_output_shape_preserved(self):
        img = np.random.rand(60, 80)
        mask = np.ones((60, 80))
        out = apply_mask(img, mask)
        self.assertEqual(out.shape, img.shape)


class TestCropRect(unittest.TestCase):

    def test_output_is_square(self):
        """Cropped output should be approximately 2*radius wide."""
        img = np.zeros((200, 200))
        loc = (100, 100, 40)
        cropped, new_loc = crop_rect(img, loc)
        self.assertEqual(cropped.shape[0], 80)
        self.assertEqual(cropped.shape[1], 80)

    def test_new_loc_radius_unchanged(self):
        """The radius in the new location should equal the original radius."""
        img = np.zeros((200, 200))
        loc = (100, 100, 40)
        _, new_loc = crop_rect(img, loc)
        self.assertEqual(new_loc[2], 40)

    def test_none_loc_returns_image_unchanged(self):
        """Passing None loc should return the original image and None."""
        img = np.zeros((100, 100))
        out_img, out_loc = crop_rect(img, None)
        np.testing.assert_array_equal(out_img, img)
        self.assertIsNone(out_loc)

    def test_clamps_to_image_boundary(self):
        """A location near the edge should not raise and should clip to image boundary."""
        img = np.zeros((100, 100))
        loc = (5, 5, 40)  # radius extends past top-left boundary
        cropped, _ = crop_rect(img, loc)
        self.assertGreater(cropped.size, 0)

    def test_content_preserved(self):
        """A bright pixel at a known location should appear in the cropped output."""
        img = np.zeros((200, 200))
        img[100, 100] = 255.0
        loc = (100, 100, 40)
        cropped, new_loc = crop_rect(img, loc)
        cx, cy = new_loc[0], new_loc[1]
        self.assertAlmostEqual(float(cropped[cy, cx]), 255.0)


if __name__ == '__main__':
    unittest.main()

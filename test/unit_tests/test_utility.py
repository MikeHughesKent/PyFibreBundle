# -*- coding: utf-8 -*-
"""
Unit tests for pybundle.utility module.

Tests cover:
    - extract_central
    - to8bit
    - to16bit
    - average_channels
    - max_channels
    - resample
    - radial_profile

"""

import unittest
import numpy as np

import context

from pybundle.utility import (
    extract_central,
    to8bit,
    to16bit,
    average_channels,
    max_channels,
    resample,
    radial_profile,
)


class TestExtractCentral(unittest.TestCase):

    def test_square_input_no_boxsize_unchanged(self):
        """Square image with no boxSize returns same-sized image."""
        img = np.zeros((100, 100))
        out = extract_central(img)
        self.assertEqual(out.shape, (100, 100))

    def test_non_square_returns_square(self):
        """Non-square input returns a square cropped image."""
        img = np.zeros((100, 200))
        out = extract_central(img)
        self.assertEqual(out.shape[0], out.shape[1])

    def test_boxsize_limits_output(self):
        """boxSize parameter limits the output dimensions."""
        img = np.zeros((200, 200))
        out = extract_central(img, boxSize=40)
        # Output should be <= 2 * boxSize on each side
        self.assertLessEqual(out.shape[0], 80)
        self.assertLessEqual(out.shape[1], 80)

    def test_content_preserved(self):
        """Pixel values in the central region are preserved."""
        img = np.ones((100, 100)) * 5.0
        img[20:80, 20:80] = 10.0
        out = extract_central(img, boxSize=20)
        # The central 40x40 region should contain 10.0
        self.assertAlmostEqual(float(out[out.shape[0] // 2, out.shape[1] // 2]), 10.0)


class TestTo8bit(unittest.TestCase):

    def test_output_dtype(self):
        img = np.array([[0.0, 128.0, 255.0]])
        out = to8bit(img)
        self.assertEqual(out.dtype, np.uint8)

    def test_full_range_mapping(self):
        """Min pixel maps to 0, max pixel maps to 255."""
        img = np.array([[0.0, 100.0]])
        out = to8bit(img)
        self.assertEqual(int(out[0, 0]), 0)
        self.assertEqual(int(out[0, 1]), 255)

    def test_explicit_min_max(self):
        """Explicit minVal and maxVal are respected."""
        img = np.array([[0.0, 50.0, 100.0]])
        out = to8bit(img, minVal=0.0, maxVal=100.0)
        self.assertEqual(int(out[0, 0]), 0)
        self.assertEqual(int(out[0, 2]), 255)

    def test_uniform_image(self):
        """Uniform image does not raise and returns valid array."""
        img = np.ones((10, 10)) * 5.0
        # Should not raise even with zero range when using explicit values
        out = to8bit(img, minVal=0.0, maxVal=10.0)
        self.assertEqual(out.dtype, np.uint8)

    def test_output_shape_preserved(self):
        img = np.random.rand(50, 60)
        out = to8bit(img)
        self.assertEqual(out.shape, (50, 60))


class TestTo16bit(unittest.TestCase):

    def test_output_dtype(self):
        img = np.array([[0.0, 1000.0, 5000.0]])
        out = to16bit(img)
        self.assertEqual(out.dtype, np.uint16)

    def test_output_shape_preserved(self):
        img = np.random.rand(50, 60) * 1000
        out = to16bit(img)
        self.assertEqual(out.shape, (50, 60))

    def test_min_pixel_maps_to_zero(self):
        """Minimum pixel value maps to 0."""
        img = np.array([[0.0, 500.0, 1000.0]])
        out = to16bit(img)
        self.assertEqual(int(out[0, 0]), 0)

    def test_monotonic_scaling(self):
        """Higher input values produce higher output values."""
        img = np.array([[0.0, 100.0, 200.0, 300.0]])
        out = to16bit(img)
        self.assertTrue(int(out[0, 0]) <= int(out[0, 1]) <= int(out[0, 2]) <= int(out[0, 3]))


class TestAverageChannels(unittest.TestCase):

    def test_2d_input_returned_unchanged(self):
        """2D input is returned as-is."""
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = average_channels(img)
        np.testing.assert_array_equal(out, img)

    def test_3d_averages_correctly(self):
        """3D image: result is mean across colour channels."""
        ch1 = np.ones((4, 4)) * 10
        ch2 = np.ones((4, 4)) * 20
        ch3 = np.ones((4, 4)) * 30
        img = np.stack([ch1, ch2, ch3], axis=2)
        out = average_channels(img)
        self.assertEqual(out.ndim, 2)
        self.assertAlmostEqual(float(out[0, 0]), 20.0)

    def test_3d_output_shape(self):
        img = np.random.rand(50, 60, 3)
        out = average_channels(img)
        self.assertEqual(out.shape, (50, 60))


class TestMaxChannels(unittest.TestCase):

    def test_2d_input_returned_unchanged(self):
        img = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = max_channels(img)
        np.testing.assert_array_equal(out, img)

    def test_3d_takes_maximum(self):
        """3D image: result is maximum across colour channels."""
        ch1 = np.ones((4, 4)) * 10
        ch2 = np.ones((4, 4)) * 50
        ch3 = np.ones((4, 4)) * 30
        img = np.stack([ch1, ch2, ch3], axis=2)
        out = max_channels(img)
        self.assertEqual(out.ndim, 2)
        self.assertAlmostEqual(float(out[0, 0]), 50.0)

    def test_3d_output_shape(self):
        img = np.random.rand(50, 60, 3)
        out = max_channels(img)
        self.assertEqual(out.shape, (50, 60))


class TestResample(unittest.TestCase):

    def test_upscale_shape(self):
        """Factor > 1 increases image dimensions."""
        img = np.zeros((100, 80), dtype=np.float32)
        out = resample(img, factor=2.0)
        self.assertEqual(out.shape, (200, 160))

    def test_downscale_shape(self):
        """Factor < 1 decreases image dimensions."""
        img = np.zeros((100, 80), dtype=np.float32)
        out = resample(img, factor=0.5)
        self.assertEqual(out.shape, (50, 40))

    def test_factor_one_preserves_shape(self):
        img = np.zeros((100, 80), dtype=np.float32)
        out = resample(img, factor=1.0)
        self.assertEqual(out.shape, (100, 80))


class TestRadialProfile(unittest.TestCase):

    def test_constant_image_constant_profile(self):
        """A uniform image should give a constant radial profile."""
        img = np.ones((50, 50)) * 7.0
        centre = (25, 25)
        profile = radial_profile(img, centre)
        # All values should be approximately 7
        np.testing.assert_allclose(profile, 7.0, rtol=1e-5)

    def test_output_is_1d(self):
        img = np.random.rand(50, 50)
        centre = (25, 25)
        profile = radial_profile(img, centre)
        self.assertEqual(profile.ndim, 1)

    def test_profile_length(self):
        """Profile length should cover radii from 0 to the corner of the image."""
        img = np.ones((50, 50))
        centre = (25, 25)
        profile = radial_profile(img, centre)
        # Length should be at least the half-width of the image
        self.assertGreaterEqual(len(profile), 25)


if __name__ == '__main__':
    unittest.main()

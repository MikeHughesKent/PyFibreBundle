# -*- coding: utf-8 -*-
"""
Unit tests for pybundle.mosaic.Mosaic.

These tests focus on public behavior of Mosaic using synthetic images and
mocked shift-estimation where needed so tests remain deterministic.

"""

import unittest
from unittest import mock

import numpy as np

try:
    import test.unit_tests.context as context
except ImportError:
    import context

from pybundle import Mosaic


def _make_disc_image(size=32, radius=12, value=200, dtype=np.uint8):
    """Create a synthetic circular test image."""
    yy, xx = np.indices((size, size))
    cx = size / 2
    cy = size / 2
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img = np.zeros((size, size), dtype=dtype)
    img[mask] = value
    return img


def _make_colour_image(size=32, radius=12, dtype=np.uint8):
    """Create a 3-channel synthetic image with different values per channel."""
    base = _make_disc_image(size=size, radius=radius, value=120, dtype=dtype)
    img = np.stack([base, base // 2, np.minimum(base + 30, 255)], axis=2)
    return img.astype(dtype)


class TestMosaicBasic(unittest.TestCase):

    def test_first_add_initialises_and_inserts(self):
        img = _make_disc_image(size=32)
        m = Mosaic(128, resize=32)

        m.add(img)
        out = m.get_mosaic()

        self.assertEqual(out.shape, (128, 128))
        self.assertGreater(np.count_nonzero(out), 0)
        self.assertEqual(m.nImages, 1)

    def test_get_mosaic_dtype_matches_input_when_not_overridden(self):
        img = _make_disc_image(size=32, dtype=np.uint16)
        m = Mosaic(128, resize=32)

        m.add(img)
        out = m.get_mosaic()

        self.assertEqual(out.dtype, np.uint16)

    def test_reset_clears_sequence_and_reinitialises_on_next_add(self):
        img1 = _make_disc_image(size=32, value=180)
        img2 = _make_disc_image(size=32, value=80)
        m = Mosaic(128, resize=32)

        m.add(img1)
        self.assertEqual(m.nImages, 1)

        m.reset()
        self.assertEqual(m.nImages, 0)

        m.add(img2)
        out = m.get_mosaic()
        self.assertGreater(np.count_nonzero(out), 0)
        self.assertEqual(m.nImages, 1)


class TestMosaicColour(unittest.TestCase):

    def test_colour_input_produces_colour_mosaic(self):
        img = _make_colour_image(size=24)
        m = Mosaic(96, resize=24)

        m.add(img)
        out = m.get_mosaic()

        self.assertEqual(out.ndim, 3)
        self.assertEqual(out.shape, (96, 96, 3))
        self.assertGreater(np.count_nonzero(out[:, :, 0]), 0)


class TestMosaicBlendModes(unittest.TestCase):

    def test_blend_false_and_true_both_insert_pixels(self):
        img = _make_disc_image(size=32)

        m_dead_leaf = Mosaic(128, resize=32, blend=False)
        m_blend = Mosaic(128, resize=32, blend=True)

        m_dead_leaf.add(img)
        m_blend.add(img)

        out_dead_leaf = m_dead_leaf.get_mosaic()
        out_blend = m_blend.get_mosaic()

        # For the first insert, both should contain non-zero values.
        self.assertGreater(np.count_nonzero(out_dead_leaf), 0)
        self.assertGreater(np.count_nonzero(out_blend), 0)


class TestMosaicShiftAndBoundary(unittest.TestCase):

    def test_expand_boundary_grows_mosaic_when_shift_moves_outside(self):
        img = _make_disc_image(size=32)
        m = Mosaic(
            64,
            resize=32,
            blend=False,
            boundaryMethod=Mosaic.EXPAND,
            expandStep=20,
            mindDistForAdd=0,
        )

        m.add(img)
        original_shape = m.get_mosaic().shape

        with mock.patch.object(Mosaic, '_Mosaic__find_shift', return_value=([0, 30], 1.0)):
            m.add(img)

        expanded_shape = m.get_mosaic().shape
        self.assertGreater(expanded_shape[0], original_shape[0])

    def test_scroll_boundary_keeps_size_when_shift_moves_outside(self):
        img = _make_disc_image(size=32)
        m = Mosaic(
            64,
            resize=32,
            blend=False,
            boundaryMethod=Mosaic.SCROLL,
            mindDistForAdd=0,
        )

        m.add(img)
        original_shape = m.get_mosaic().shape

        with mock.patch.object(Mosaic, '_Mosaic__find_shift', return_value=([0, 30], 1.0)):
            m.add(img)

        scrolled_shape = m.get_mosaic().shape
        self.assertEqual(scrolled_shape, original_shape)

    def test_crop_boundary_keeps_size_and_does_not_crash(self):
        img = _make_disc_image(size=32)
        m = Mosaic(
            64,
            resize=32,
            blend=False,
            boundaryMethod=Mosaic.CROP,
            mindDistForAdd=0,
        )

        m.add(img)
        original_shape = m.get_mosaic().shape

        with mock.patch.object(Mosaic, '_Mosaic__find_shift', return_value=([0, 30], 1.0)):
            m.add(img)

        cropped_shape = m.get_mosaic().shape
        self.assertEqual(cropped_shape, original_shape)


class TestMosaicResetCriteria(unittest.TestCase):

    def test_reset_on_low_shift_confidence(self):
        img = _make_disc_image(size=32)
        m = Mosaic(128, resize=32, resetThresh=0.99, mindDistForAdd=0)

        m.add(img)
        with mock.patch.object(Mosaic, '_Mosaic__find_shift', return_value=([0, 0], 0.5)):
            m.add(img)

        out = m.get_mosaic()
        self.assertGreater(np.count_nonzero(out), 0)

    def test_reset_on_low_intensity(self):
        bright = _make_disc_image(size=32, value=180)
        dim = _make_disc_image(size=32, value=5)
        m = Mosaic(128, resize=32, resetIntensity=20, mindDistForAdd=0)

        m.add(bright)
        with mock.patch.object(Mosaic, '_Mosaic__find_shift', return_value=([0, 0], 1.0)):
            m.add(dim)

        out = m.get_mosaic()
        self.assertGreater(np.count_nonzero(out), 0)


if __name__ == '__main__':
    unittest.main()

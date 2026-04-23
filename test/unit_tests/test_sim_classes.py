# -*- coding: utf-8 -*-
"""Unit tests for simulation classes (CorePacking and BundleSim)."""

import unittest
import numpy as np

import context

from pybundle.sim import BundleSim, regular_core_lattice
from pybundle.core_interpolation import find_cores
from pybundle.sim_core_pos import CorePacking


class TestCorePacking(unittest.TestCase):

    def test_core_positions_span_bundle_radius_before_and_after_run(self):
        bundle_radius = 40
        sim = CorePacking(
            n_cores=4000,
            bundle_radius=bundle_radius,
            radius_mean=1.0,
            radius_std=0.01,
            noise=0.0,
            seed=7,
            debug_timing=False,
        )

        pos_before, _ = sim.cores()
        self.assertAlmostEqual(abs(float(np.min(pos_before[:, 0]))), bundle_radius, delta=3.0)
        self.assertAlmostEqual(float(np.max(pos_before[:, 0])), bundle_radius, delta=3.0)
        self.assertAlmostEqual(abs(float(np.min(pos_before[:, 1]))), bundle_radius, delta=3.0)
        self.assertAlmostEqual(float(np.max(pos_before[:, 1])), bundle_radius, delta=3.0)

        pos_after, _ = sim.run(n_steps=5)
        self.assertAlmostEqual(abs(float(np.min(pos_after[:, 0]))), bundle_radius, delta=3.0)
        self.assertAlmostEqual(float(np.max(pos_after[:, 0])), bundle_radius, delta=3.0)
        self.assertAlmostEqual(abs(float(np.min(pos_after[:, 1]))), bundle_radius, delta=3.0)
        self.assertAlmostEqual(float(np.max(pos_after[:, 1])), bundle_radius, delta=3.0)

    def test_init_and_run_return_expected_shapes(self):
        sim = CorePacking(
            n_cores=200,
            bundle_radius=40,
            radius_mean=1.0,
            radius_std=0.01,
            noise=0.0,
            seed=1,
            debug_timing=False,
        )

        self.assertEqual(sim.x.shape, (200, 2))
        self.assertEqual(sim.r.shape, (200,))
        self.assertEqual(sim.step_count, 0)

        pos, rad = sim.run(n_steps=2)
        self.assertEqual(pos.shape, (200, 2))
        self.assertEqual(rad.shape, (200,))
        self.assertEqual(sim.step_count, 2)

    def test_nearest_core_distance_image_fixed_size_masks_outside_bundle(self):
        sim = CorePacking(
            n_cores=150,
            bundle_radius=30,
            radius_mean=1.0,
            radius_std=0.01,
            noise=0.0,
            seed=2,
            debug_timing=False,
        )

        img, xx, yy = sim._nearest_core_distance_image(img_size=81)

        self.assertEqual(img.shape, (81, 81))
        self.assertEqual(xx.shape, (81, 81))
        self.assertEqual(yy.shape, (81, 81))

        rr = np.sqrt(xx**2 + yy**2)
        outside = rr > sim.R
        self.assertTrue(np.all(img[outside] == 0.0))

    def test_nearest_core_distance_image_auto_size(self):
        sim = CorePacking(
            n_cores=120,
            bundle_radius=25,
            radius_mean=1.0,
            radius_std=0.01,
            noise=0.0,
            seed=3,
            debug_timing=False,
        )

        img, xx, yy = sim._nearest_core_distance_image(img_size=None)

        self.assertEqual(img.ndim, 2)
        self.assertEqual(img.shape, xx.shape)
        self.assertEqual(img.shape, yy.shape)
        self.assertGreaterEqual(img.shape[0], 64)


class TestBundleSim(unittest.TestCase):

    def _build_regular_bundle(self):
        core_x, core_y = regular_core_lattice(
            spacing=4,
            radius=100,
            packing="hex",
            shape="circle",
        )
        core_r = np.full(core_x.shape, 1.2, dtype=float)
        return core_x, core_y, core_r

    def test_ref_image_inside_frame_has_signal(self):
        core_x, core_y, core_r = self._build_regular_bundle()

        sim = BundleSim(
            core_x=core_x,
            core_y=core_y,
            core_radius=core_r,
            img_size=(128, 128),
            bundle_offset=(0, 0),
            pixel_size=1.0,
        )
        img = sim.ref_image()

        self.assertEqual(img.shape, (128, 128))
        self.assertGreater(float(np.sum(img)), 0.0)

    def test_ref_image_partial_outside_frame_has_reduced_signal(self):
        core_x, core_y, core_r = self._build_regular_bundle()

        sim_inside = BundleSim(
            core_x=core_x,
            core_y=core_y,
            core_radius=core_r,
            img_size=(128, 128),
            bundle_offset=(0, 0),
            pixel_size=1.0,
        )
        img_inside = sim_inside.ref_image()

        sim_partial = BundleSim(
            core_x=core_x,
            core_y=core_y,
            core_radius=core_r,
            img_size=(128, 128),
            bundle_offset=(42, 0),
            pixel_size=1.0,
        )
        img_partial = sim_partial.ref_image()

        self.assertGreater(float(np.sum(img_partial)), 0.0)
        self.assertLess(float(np.sum(img_partial)), float(np.sum(img_inside)))

    def test_ref_image_entire_bundle_outside_frame_is_zero(self):
        core_x, core_y, core_r = self._build_regular_bundle()

        sim = BundleSim(
            core_x=core_x,
            core_y=core_y,
            core_radius=core_r,
            img_size=(128, 128),
            bundle_offset=(400, 0),
            pixel_size=1.0,
        )
        img = sim.ref_image()

        self.assertTrue(np.allclose(img, 0.0))

    def test_find_cores_recovers_generated_core_count(self):
        core_x, core_y = regular_core_lattice(
            spacing=6,
            radius=24,
            packing="hex",
            shape="circle",
        )
        core_r = np.full(core_x.shape, 1.2, dtype=float)

        sim = BundleSim(
            core_x=core_x,
            core_y=core_y,
            core_radius=core_r,
            img_size=(160, 160),
            bundle_offset=(0, 0),
            pixel_size=1.0,
        )
        img = sim.ref_image()

        found_x, found_y = find_cores(img, coreSpacing=6)

        self.assertEqual(len(found_x), len(core_x))
        self.assertEqual(len(found_y), len(core_y))


if __name__ == "__main__":
    unittest.main()

# -*- coding: utf-8 -*-
"""
Example of analysing fibre core neighbour statistics from calibration images.

This script loads example calibration images, finds core centres, runs
neighbour-statistics analysis, prints key metrics, and displays diagnostic
plots.
"""

from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import context

from pybundle import auto_mask, find_core_spacing, find_cores, core_pattern_statistics


bundle_img =  Path('../test/data/usaf1_background.tif')


def _summarise_stats(stats):
    print(f"  cores detected:   {stats['num_input_cores']}")
    print(f"  cores analysed:   {stats['num_cores']}")
    print(f"  boundary cores:   {stats['num_boundary_cores']}")
    print(f"  mean NN distance: {stats['mean_nearest_neighbour_dist']:.3f}")
    print(f"  NN distance CV:   {stats['cv_nearest_neighbour_dist']:.4f}")
    print(f"  mean |psi6|:      {stats['mean_local_psi6_mag']:.4f}")
    print(f"  global |psi6|:    {stats['global_psi6_mag']:.4f}")
    print(f"  defect fraction:  {stats['defect_fraction']:.4f}")
    print(f"  coord hist:       {stats['coord_hist']}")


# Load image
img = np.array(Image.open(bundle_img))

# Make sure we don't end up with spurious core detections
img = auto_mask(img)


# Locate cores
core_spacing = find_core_spacing(img)
core_x, core_y = find_cores(img, core_spacing)

# Calculate stats
stats = core_pattern_statistics(
    core_x, core_y, k_neighbours=6, boundary_layers = 2)


# Display key results
print(f"\nAnalysis for {bundle_img.name}")
_summarise_stats(stats)

# Display bundle image and histograms of nearest neighbour distances
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

ax0 = axes[0]
ax0.imshow(img, cmap='gray')
ax0.plot(core_x[stats['analysis_mask']],
         core_y[stats['analysis_mask']], 'c.', markersize=1.5)
ax0.plot(core_x[stats['boundary_mask']],
         core_y[stats['boundary_mask']], 'r.', markersize=2)
ax0.set_title(f"{bundle_img.name}: analysed and boundary cores")
ax0.set_axis_off()

ax1 = axes[1]

dists = stats['nearest_neighbour_dists']
ax1.hist(stats['nearest_neighbour_dists'], bins=40)
ax1.set_title('Nearest-neighbour distance distribution')
ax1.set_xlabel('Distance (pixels)')
ax1.set_ylabel('Count')

plt.tight_layout()

plt.show()

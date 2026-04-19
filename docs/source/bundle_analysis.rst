================
Bundle Analysis
================

PyFibreBundle provides tools to analyse the spatial arrangement and packing geometry of fibre cores within a bundle. This is useful for understanding bundle quality, measuring local disorder, and detecting structural defects.

Cores can be localised using the :py:func:`pybundle.find_cores` function, which returns their (x, y) coordinates. The :py:func:`pybundle.core_pattern_statistics` function then takes these coordinates as input to perform a detailed analysis of the core pattern.

For example, assuming you have a calibration image of your bundle, you can run:

    import pybundle as pyb
    core_spacing = pyb.find_core_spacing(calib_img)
    core_x, core_y = pyb.find_cores(calib_img, core_spacing)
    stats = pyb.core_pattern_statistics(core_x, core_y)

This analyses the core pattern for:

* **Local packing order**: Uses the complex bond-order parameter ψ₆ to measure how close each core neighbourhood is to ideal hexagonal packing.
* **Spacing uniformity**: Reports nearest-neighbour distance statistics (mean, standard deviation, and coefficient of variation).
* **Topological defects**: Identifies cores with non-6 coordination numbers 
* **Global disorder**: Computes the magnitude of the globally-averaged bond-order parameter as a large-scale disorder indicator.

The function returns a dictionary with:

* Scalar summaries: ``num_cores``, ``mean_nearest_neighbour_dist``, ``cv_nearest_neighbour_dist``, ``mean_local_psi6_mag``, ``global_psi6_mag``, ``defect_fraction``
* Per-core arrays: ``nearest_neighbour_distances``, ``local_psi6``, ``local_psi6_mag``, ``coord_num``
* Masks: ``analysis_mask`` and ``boundary_mask`` indicating which cores were included or excluded

Statistics are biased by cores on the bundle boundary, which have incomplete neighbourhoods. The function uses an alpha-shape concave-hull algorithm to identify and exclude boundary cores, optionally removing multiple layers of Delaunay neighbours. This improves the quality of interior-core statistics and makes results more representative of true bundle packing quality.

Example
^^^^^^^

See the `Analysis Example <https://github.com/MikeHughesKent/PyFibreBundle/tree/main/examples/analysis_example.py>`_ for a complete worked example that:



# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This module contains code for simulating fibre bundle images. 
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from PIL import Image

from scipy.spatial import cKDTree


def _resolve_rng(seed=None, rng=None):
    """Return a NumPy random number generator.

    If rng is provided it is returned unchanged; otherwise a new generator
    is created from seed.
    """
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _sample_field_bilinear(field, x_axis, y_axis, sample_x, sample_y):
    """Sample a 2D field on regular axes at arbitrary points via bilinear interpolation."""
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]

    fx = (sample_x - x_axis[0]) / dx
    fy = (sample_y - y_axis[0]) / dy

    ix0 = np.floor(fx).astype(int)
    iy0 = np.floor(fy).astype(int)
    ix0 = np.clip(ix0, 0, len(x_axis) - 2)
    iy0 = np.clip(iy0, 0, len(y_axis) - 2)

    tx = fx - ix0
    ty = fy - iy0

    f00 = field[iy0, ix0]
    f01 = field[iy0, ix0 + 1]
    f10 = field[iy0 + 1, ix0]
    f11 = field[iy0 + 1, ix0 + 1]

    return (1 - tx) * (1 - ty) * f00 + tx * (1 - ty) * f01 + (1 - tx) * ty * f10 + tx * ty * f11


def _correlated_random_field(x_axis, y_axis, corr_length, rng):
    """Generate a zero-mean unit-std correlated random field on a regular grid."""
    ny = len(y_axis)
    nx = len(x_axis)
    dx = x_axis[1] - x_axis[0]
    dy = y_axis[1] - y_axis[0]

    noise = rng.normal(0.0, 1.0, size=(ny, nx))

    kx = np.fft.fftfreq(nx, d=dx)
    ky = np.fft.fftfreq(ny, d=dy)
    kx2, ky2 = np.meshgrid(kx, ky)
    k2 = kx2**2 + ky2**2

    # Gaussian low-pass in k-space gives smooth, spatially-correlated displacement.
    filt = np.exp(-2.0 * (math.pi**2) * (corr_length**2) * k2)

    field = np.fft.ifft2(np.fft.fft2(noise) * filt).real
    field = field - np.mean(field)
    field_std = np.std(field)
    if field_std > 0:
        field = field / field_std

    return field


def _apply_correlated_displacement(coreX, coreY, spacing, disp_sd, corr_length, rng):
    """Apply a smooth displacement field to core positions."""
    if disp_sd <= 0:
        return coreX, coreY

    if corr_length is None:
        corr_length = 10 * spacing

    extent = max(np.max(np.abs(coreX)), np.max(np.abs(coreY))) + 2 * spacing
    grid_step = max(spacing / 2, 1e-6)

    nx = int(np.ceil((2 * extent) / grid_step)) + 1
    ny = nx

    x_axis = np.linspace(-extent, extent, nx)
    y_axis = np.linspace(-extent, extent, ny)

    field_x = _correlated_random_field(x_axis, y_axis, corr_length, rng)
    field_y = _correlated_random_field(x_axis, y_axis, corr_length, rng)

    disp_x = _sample_field_bilinear(field_x, x_axis, y_axis, coreX, coreY)
    disp_y = _sample_field_bilinear(field_y, x_axis, y_axis, coreX, coreY)

    coreX = coreX + disp_x * disp_sd
    coreY = coreY + disp_y * disp_sd
    return coreX, coreY


def regular_lattice_centres(spacing=1, radius=100, packing='hex'):
    """Generate regular square or hex lattice core centres inside a bounding square.
    
    Keyword Arguments:
        spacing : float
                  centre-centre core spacing (default = 1)                  
        radius  : float
                  bundle radius, same units as spacing (default = 100)
        packing : str
                  core packing, 'hex' (default) or 'square'

        Returns:    
        coreX, coreY: 1D arrays of x and y coordinates of core centres, in same units as spacing     
    """
    
    assert packing in ['hex', 'square'], "packing must be 'hex' or 'square'"
        
    numX = int(radius / spacing) * 2 + 1

    if packing == 'hex':
        numY = round(numX * 2 / math.sqrt(3))
    else:
        numY = numX

    coreX, coreY = np.meshgrid(
        np.linspace(-radius, radius, numX),
        np.linspace(-radius, radius, numY)
    )

    if packing == 'hex':
        coreX[1::2] = coreX[1::2] + float(spacing) / 2

    coreX = np.squeeze(np.reshape(coreX, [coreX.size, 1]))
    coreY = np.squeeze(np.reshape(coreY, [coreY.size, 1]))

    return coreX, coreY


def apply_random_nudges(coreX, coreY, position_sd=0, seed=None, rng=None):
    """Apply uncorrelated Gaussian position nudges to core centres."""
    if position_sd <= 0:
        return coreX, coreY

    rng = _resolve_rng(seed=seed, rng=rng)
    coreX = coreX + rng.normal(loc=0, scale=position_sd, size=np.shape(coreX)[0])
    coreY = coreY + rng.normal(loc=0, scale=position_sd, size=np.shape(coreX)[0])
    return coreX, coreY


def apply_correlated_shifts(coreX, coreY, spacing=1, disp_sd=0, corr_length=None, seed=None, rng=None):
    """Apply smooth spatially-correlated shifts to core centres."""
    if disp_sd <= 0:
        return coreX, coreY

    rng = _resolve_rng(seed=seed, rng=rng)
    return _apply_correlated_displacement(
        coreX,
        coreY,
        spacing=spacing,
        disp_sd=disp_sd,
        corr_length=corr_length,
        rng=rng
    )


def _clip_core_positions(coreX, coreY, radius=100, shape='circle'):
    """Clip core centres to requested bundle shape."""
    if shape == 'circle':
        coreRadSq = coreX**2 + coreY**2
        coreX = np.delete(coreX, coreRadSq > radius**2)
        coreY = np.delete(coreY, coreRadSq > radius**2)
    return coreX, coreY


def _stack_cores_from_exclusion_circles(coreX, coreY, exclusion_radii, rng, drift_gain=1.0, passes=2):
    """Stack cores sequentially using per-core exclusion circles.

    Earlier cores are treated as fixed anchors. Later cores are moved away from
    overlaps, so local radius variation accumulates into smooth spatial drift.
    """
    coreX = np.asarray(coreX, dtype=float).copy()
    coreY = np.asarray(coreY, dtype=float).copy()
    exclusion_radii = np.asarray(exclusion_radii, dtype=float)

    order = np.lexsort((coreX, coreY))
    placed = []

    for idx in order:
        x_i = coreX[idx]
        y_i = coreY[idx]
        r_i = exclusion_radii[idx]

        for _ in range(max(1, int(passes))):
            moved = False
            for j in placed:
                dx = x_i - coreX[j]
                dy = y_i - coreY[j]
                dist = math.hypot(dx, dy)
                min_dist = r_i + exclusion_radii[j]

                if dist < min_dist:
                    if dist < 1e-12:
                        theta = rng.uniform(0.0, 2.0 * math.pi)
                        dx = math.cos(theta)
                        dy = math.sin(theta)
                        dist = 1.0

                    shift = (min_dist - dist) * drift_gain
                    x_i = x_i + shift * dx / dist
                    y_i = y_i + shift * dy / dist
                    moved = True

            if not moved:
                break

        coreX[idx] = x_i
        coreY[idx] = y_i
        placed.append(idx)

    return coreX, coreY


def apply_stacked_circle_drift(
        coreX,
        coreY,
        spacing=1,
        radius_sd=0.15,
        radius_corr_length=None,
        drift_gain=1.0,
        stack_passes=2,
        seed=None,
        rng=None):
    """Create drift by stacking cores with variable exclusion-circle radii."""
    if radius_sd <= 0:
        return np.asarray(coreX, dtype=float), np.asarray(coreY, dtype=float)

    rng = _resolve_rng(seed=seed, rng=rng)
    coreX = np.asarray(coreX, dtype=float)
    coreY = np.asarray(coreY, dtype=float)

    if radius_corr_length is None:
        radius_corr_length = 12 * spacing

    extent = max(np.max(np.abs(coreX)), np.max(np.abs(coreY))) + 2 * spacing
    grid_step = max(spacing / 2, 1e-6)
    nx = int(np.ceil((2 * extent) / grid_step)) + 1
    ny = nx

    x_axis = np.linspace(-extent, extent, nx)
    y_axis = np.linspace(-extent, extent, ny)
    radius_field = _correlated_random_field(x_axis, y_axis, radius_corr_length, rng)
    radius_noise = _sample_field_bilinear(radius_field, x_axis, y_axis, coreX, coreY)

    base_radius = spacing / 2.0
    exclusion_radii = base_radius * (1.0 + radius_sd * radius_noise)
    exclusion_radii = np.clip(exclusion_radii, 0.2 * base_radius, 3.0 * base_radius)

    return _stack_cores_from_exclusion_circles(
        coreX,
        coreY,
        exclusion_radii,
        rng=rng,
        drift_gain=drift_gain,
        passes=stack_passes
    )


def _points_in_boundary(points, domain_size, boundary=None):
    """Return boolean mask of points that lie inside domain and optional boundary."""
    width, height = float(domain_size[0]), float(domain_size[1])
    points = np.asarray(points, dtype=float)
    in_rect = (
        (points[:, 0] >= 0.0)
        & (points[:, 0] <= width)
        & (points[:, 1] >= 0.0)
        & (points[:, 1] <= height)
    )

    if boundary is None:
        return in_rect

    if callable(boundary):
        try:
            extra = np.asarray(boundary(points), dtype=bool)
            if extra.shape != (points.shape[0],):
                raise ValueError
        except Exception:
            try:
                extra = np.asarray(boundary(points[:, 0], points[:, 1]), dtype=bool)
            except Exception:
                extra = np.asarray([bool(boundary(p[0], p[1])) for p in points], dtype=bool)
        return in_rect & extra

    if isinstance(boundary, np.ndarray):
        if boundary.ndim != 2:
            raise ValueError("boundary mask must be 2D when provided as ndarray")
        mask_h, mask_w = boundary.shape
        x_idx = np.clip((points[:, 0] / max(width, 1e-12) * (mask_w - 1)).astype(int), 0, mask_w - 1)
        y_idx = np.clip((points[:, 1] / max(height, 1e-12) * (mask_h - 1)).astype(int), 0, mask_h - 1)
        return in_rect & boundary[y_idx, x_idx].astype(bool)

    raise ValueError("boundary must be None, callable, or a 2D ndarray mask")


def _nearest_seed_indices(points, seeds):
    """Return nearest-seed index for each point."""
    if points.shape[0] == 0:
        return np.empty(0, dtype=int)
    diff = points[:, None, :] - seeds[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.argmin(dist2, axis=1)


def _pair_indices_within_cutoff(points, cutoff):
    """Return unordered point index pairs whose distance is below cutoff."""
    n_pts = points.shape[0]
    if n_pts < 2:
        return np.empty((0, 2), dtype=int)

    if cKDTree is not None:
        tree = cKDTree(points)
        pairs = np.array(list(tree.query_pairs(cutoff)), dtype=int)
        if pairs.size == 0:
            return np.empty((0, 2), dtype=int)
        return pairs

    idx_i, idx_j = np.triu_indices(n_pts, k=1)
    vec = points[idx_i] - points[idx_j]
    dist2 = np.sum(vec * vec, axis=1)
    keep = dist2 < cutoff * cutoff
    return np.column_stack((idx_i[keep], idx_j[keep]))


def relax_fused_bundle_core_positions(
        positions,
        core_spacing,
        relax_steps=120,
        dt=0.01,
        cutoff=None,
        damping=0.15,
        domain_size=None,
        boundary=None,
        neighbour_rebuild=10):
    """Relax core positions using short-range repulsive forces."""
    
    positions = np.asarray(positions, dtype=float).reshape(-1, 2).copy()
    if positions.shape[0] == 0 or relax_steps <= 0:
        return positions

    if cutoff is None:
        cutoff = 2.0 * core_spacing

    prev_disp = np.zeros_like(positions)
    neighbour_rebuild = max(1, int(neighbour_rebuild))
    pair_cache = None

    for step in range(int(relax_steps)):
        if pair_cache is None or (step % neighbour_rebuild == 0):
            pair_cache = _pair_indices_within_cutoff(positions, cutoff)

        forces = np.zeros_like(positions)

        if pair_cache.shape[0] > 0:
            i_idx = pair_cache[:, 0]
            j_idx = pair_cache[:, 1]
            vec = positions[i_idx] - positions[j_idx]
            dist = np.linalg.norm(vec, axis=1)

            valid = dist > 1e-12
            if np.any(valid):
                i_idx = i_idx[valid]
                j_idx = j_idx[valid]
                vec = vec[valid]
                dist = dist[valid]

                repulsion = (core_spacing / dist) ** 12
                f_ij = (vec / dist[:, None]) * repulsion[:, None]
                np.add.at(forces, i_idx, f_ij)
                np.add.at(forces, j_idx, -f_ij)

        if domain_size is not None:
            width, height = float(domain_size[0]), float(domain_size[1])
            margin = 0.5 * core_spacing
            eps = 1e-9

            left = np.clip(margin - positions[:, 0], 0.0, None)
            right = np.clip(positions[:, 0] - (width - margin), 0.0, None)
            bottom = np.clip(margin - positions[:, 1], 0.0, None)
            top = np.clip(positions[:, 1] - (height - margin), 0.0, None)

            forces[:, 0] += (left / (margin + eps)) ** 3
            forces[:, 0] -= (right / (margin + eps)) ** 3
            forces[:, 1] += (bottom / (margin + eps)) ** 3
            forces[:, 1] -= (top / (margin + eps)) ** 3

        disp = dt * forces + damping * prev_disp
        positions += disp
        prev_disp = disp

        if domain_size is not None:
            positions[:, 0] = np.clip(positions[:, 0], 0.0, float(domain_size[0]))
            positions[:, 1] = np.clip(positions[:, 1], 0.0, float(domain_size[1]))

    if domain_size is not None:
        keep = _points_in_boundary(positions, domain_size, boundary=boundary)
        positions = positions[keep]

    return positions


def grain_relaxed_bundle_centres(
        domain_size,
        core_spacing,
        n_grains,
        relax_steps=120,
        dt=0.01,
        noise_sigma=None,
        boundary=None,
        seed=None,
        rng=None,
        damping=0.15,
        neighbour_rebuild=10):
    """Generate fused-bundle core centres with grains, relaxation, and noise.

    Parameters
    ----------
    domain_size : tuple(float, float)
        Width and height of generation domain.
    core_spacing : float
        Target nearest-neighbour spacing.
    n_grains : int
        Number of Voronoi-like grains controlling patch size.
    relax_steps : int
        Number of particle relaxation iterations.
    dt : float
        Relaxation update step size.
    noise_sigma : float or None
        Gaussian positional noise sigma. Defaults to 0.01 * core_spacing.
    boundary : callable or 2D ndarray mask or None
        Optional boundary filter.
    seed : int or None
        Seed for deterministic output when rng is not provided.
    rng : np.random.Generator or None
        Optional random generator.
    damping : float
        Fraction of previous displacement added each step.
    neighbour_rebuild : int
        Rebuild neighbour structure every N relaxation steps.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) of core centre positions.
    """
    rng = _resolve_rng(seed=seed, rng=rng)
    width, height = float(domain_size[0]), float(domain_size[1])

    if core_spacing <= 0:
        raise ValueError("core_spacing must be > 0")

    n_grains = max(1, int(n_grains))
    if noise_sigma is None:
        noise_sigma = 0.01 * core_spacing

    grain_seeds = np.column_stack((
        rng.uniform(0.0, width, size=n_grains),
        rng.uniform(0.0, height, size=n_grains)
    ))
    grain_angles = rng.uniform(0.0, 2.0 * math.pi, size=n_grains)

    # Build a global index range large enough to cover each rotated grain domain.
    diag = math.hypot(width, height) + 4.0 * core_spacing
    n_i = int(np.ceil(diag / core_spacing)) + 2
    n_j = int(np.ceil(diag / (math.sqrt(3.0) * core_spacing / 2.0))) + 2

    i_vals = np.arange(-n_i, n_i + 1)
    j_vals = np.arange(-n_j, n_j + 1)
    ii, jj = np.meshgrid(i_vals, j_vals, indexing='ij')
    ii = ii.reshape(-1)
    jj = jj.reshape(-1)

    a1 = np.array([core_spacing, 0.0])
    a2 = np.array([0.5 * core_spacing, 0.5 * math.sqrt(3.0) * core_spacing])
    lattice_vectors = ii[:, None] * a1[None, :] + jj[:, None] * a2[None, :]

    pad = 2.0 * core_spacing
    pieces = []
    for g_idx in range(n_grains):
        theta = grain_angles[g_idx]
        c_t = math.cos(theta)
        s_t = math.sin(theta)
        rot = np.array([[c_t, -s_t], [s_t, c_t]])

        # Random phase per grain avoids unrealistically coherent boundaries.
        phase = rng.uniform(-0.5, 0.5, size=2) * core_spacing
        pts = grain_seeds[g_idx] + phase + lattice_vectors @ rot.T

        in_padded = (
            (pts[:, 0] >= -pad)
            & (pts[:, 0] <= width + pad)
            & (pts[:, 1] >= -pad)
            & (pts[:, 1] <= height + pad)
        )
        pts = pts[in_padded]
        if pts.shape[0] == 0:
            continue

        owner = _nearest_seed_indices(pts, grain_seeds)
        pts = pts[owner == g_idx]
        if pts.shape[0] > 0:
            pieces.append(pts)

    if len(pieces) == 0:
        return np.empty((0, 2), dtype=float)

    positions = np.vstack(pieces)
    keep = _points_in_boundary(positions, domain_size, boundary=boundary)
    positions = positions[keep]

    positions = relax_fused_bundle_core_positions(
        positions,
        core_spacing=core_spacing,
        relax_steps=relax_steps,
        dt=dt,
        cutoff=2.0 * core_spacing,
        damping=damping,
        domain_size=domain_size,
        boundary=boundary,
        neighbour_rebuild=neighbour_rebuild
    )

    if positions.shape[0] > 0 and noise_sigma > 0:
        # Increase noise near Voronoi boundaries using nearest-two-seed distance gap.
        diff = positions[:, None, :] - grain_seeds[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        if n_grains > 1:
            nearest_two = np.partition(dist2, 1, axis=1)[:, :2]
            d1 = np.sqrt(nearest_two[:, 0])
            d2 = np.sqrt(nearest_two[:, 1])
            boundary_weight = np.exp(-(d2 - d1) / max(0.35 * core_spacing, 1e-9))
        else:
            boundary_weight = np.zeros(positions.shape[0], dtype=float)

        sigma = noise_sigma * (1.0 + 0.75 * boundary_weight)
        positions += rng.normal(0.0, 1.0, size=positions.shape) * sigma[:, None]

    keep = _points_in_boundary(positions, domain_size, boundary=boundary)
    return positions[keep]


def _sample_random_points_in_boundary(domain_size, n_points, boundary, rng):
    """Sample random points inside rectangular domain and optional boundary."""
    width, height = float(domain_size[0]), float(domain_size[1])
    n_points = int(n_points)
    if n_points <= 0:
        return np.empty((0, 2), dtype=float)

    points = []
    n_collected = 0
    # Rejection sampling with adaptive oversampling.
    for _ in range(80):
        batch_size = max(64, int((n_points - n_collected) * 3.0))
        cand = np.column_stack((
            rng.uniform(0.0, width, size=batch_size),
            rng.uniform(0.0, height, size=batch_size)
        ))
        keep = _points_in_boundary(cand, domain_size, boundary=boundary)
        accepted = cand[keep]
        if accepted.shape[0] > 0:
            points.append(accepted)
            n_collected += accepted.shape[0]
            if n_collected >= n_points:
                break

    if n_collected < n_points:
        raise ValueError("could not sample enough points inside boundary")

    pts = np.vstack(points)
    return pts[:n_points]


def random_relaxed_bundle_centres(
        domain_size = (100,100),
        n_cores=None,
        core_spacing=None,
        relax_steps=120,
        dt=0.01,
        centre_pull=0.05,
        cutoff=None,
        damping=0.12,
        noise_sigma=0.0,
        boundary=None,
        seed=None,
        rng=None,
        neighbour_rebuild=10):
    """Generate core centres by random placement then relaxation.

    Cores are initialized randomly, then iteratively moved using:
    1) repulsion from nearby cores,
    2) attraction toward domain centre,
    3) optional small Gaussian noise at the end.

    If n_cores is None, it is estimated from domain area.

    """
    rng = _resolve_rng(seed=seed, rng=rng)
    width, height = float(domain_size[0]), float(domain_size[1])

    if core_spacing is None:
        raise ValueError("core_spacing must be provided")

    if n_cores is None:
       n_cores = predict_num_cores(
           spacing=core_spacing,
           radius=int(0.5 * math.hypot(width, height)),
           shape='square',
           packing='hex'
       )

    n_cores = int(n_cores)
    if n_cores <= 0:
        return np.empty((0, 2), dtype=float)
    if core_spacing <= 0:
        raise ValueError("core_spacing must be > 0")
    if cutoff is None:
        cutoff = 2.0 * core_spacing

    positions = _sample_random_points_in_boundary(
        domain_size=domain_size,
        n_points=n_cores,
        boundary=boundary,
        rng=rng
    )

    centre = np.array([0.5 * width, 0.5 * height], dtype=float)

    prev_disp = np.zeros_like(positions)
    neighbour_rebuild = max(1, int(neighbour_rebuild))
    pair_cache = None

    for step in range(int(relax_steps)):
        if pair_cache is None or (step % neighbour_rebuild == 0):
            pair_cache = _pair_indices_within_cutoff(positions, cutoff)

        forces = np.zeros_like(positions)

        # Pairwise repulsion.
        if pair_cache.shape[0] > 0:
            i_idx = pair_cache[:, 0]
            j_idx = pair_cache[:, 1]
            vec = positions[i_idx] - positions[j_idx]
            dist = np.linalg.norm(vec, axis=1)

            valid = dist > 1e-12
            if np.any(valid):
                i_idx = i_idx[valid]
                j_idx = j_idx[valid]
                vec = vec[valid]
                dist = dist[valid]

                repulsion = (core_spacing / dist) ** 12
                f_ij = (vec / dist[:, None]) * repulsion[:, None]
                np.add.at(forces, i_idx, f_ij)
                np.add.at(forces, j_idx, -f_ij)

        # Gentle spring-like pull toward domain centre.
        forces += centre_pull * (centre[None, :] - positions) / max(core_spacing, 1e-12)

        # Keep points away from hard rectangular edges to prevent pileup.
        margin = 0.5 * core_spacing
        eps = 1e-9
        left = np.clip(margin - positions[:, 0], 0.0, None)
        right = np.clip(positions[:, 0] - (width - margin), 0.0, None)
        bottom = np.clip(margin - positions[:, 1], 0.0, None)
        top = np.clip(positions[:, 1] - (height - margin), 0.0, None)
        #forces[:, 0] += (left / (margin + eps)) ** 3
        #forces[:, 0] -= (right / (margin + eps)) ** 3
        #forces[:, 1] += (bottom / (margin + eps)) ** 3
        #forces[:, 1] -= (top / (margin + eps)) ** 3

        disp = dt * forces + damping * prev_disp
        positions += disp
        prev_disp = disp

        positions[:, 0] = np.clip(positions[:, 0], 0.0, width)
        positions[:, 1] = np.clip(positions[:, 1], 0.0, height)

        if boundary is not None:
            keep = _points_in_boundary(positions, domain_size, boundary=boundary)
            n_bad = np.count_nonzero(~keep)
            if n_bad > 0:
                positions[~keep] = _sample_random_points_in_boundary(
                    domain_size=domain_size,
                    n_points=n_bad,
                    boundary=boundary,
                    rng=rng
                )
                prev_disp[~keep] = 0.0

    if noise_sigma > 0:
        positions += rng.normal(0.0, noise_sigma, size=positions.shape)

    keep = _points_in_boundary(positions, domain_size, boundary=boundary)
    return positions[keep]


def generate_fused_bundle_core_positions(*args, **kwargs):
    """Backward-compatible alias for grain_relaxed_bundle_centres."""
    return grain_relaxed_bundle_centres(*args, **kwargs)


def predict_num_cores(spacing = 1, radius = 100, shape = 'circle', packing = 'hex'):
    """ Returns predicted number of cores a simulated bundle will have.
    
    Keyword Arguments
        spacing : float
                  centre-centre core spacing (default = 1)                  
        radius  : float
                  bundle radius, same units as spacing (default = 100)
        shape   : str
                  bundle shape, 'circle' (default) or 'square'  
        packing : str
                  core packing, 'hex' (default) or 'square'          
    """       

    assert shape in ['circle', 'square'], "shape must be 'circle' or 'square'"
    assert packing in ['hex', 'square'], "packing must be 'hex' or 'square'"
           
    numCores = (2 * radius / spacing) **2
    
    if shape == 'circle':
        numCores = numCores / (radius * 2)**2 * math.pi * radius**2
     
    if packing == 'hex':
        numCores = numCores * 2 /  math.sqrt(3)
     
    return numCores


def predict_radius(spacing = 1, numCores = 30000, shape = 'circle', packing = 'hex'):
    """ Returns predicted radius required to produce a simulated bundle
    with the specified number of cores.
    
    Keyword Arguments
        spacing : float
                  centre-centre core spacing (default = 1)                  
        numCores: int
                  number of cores (default = 30000)
        shape   : str
                  bundle shape, 'circle' (default) or 'square
        packing : str
                  core packing, 'hex' (default) or 'square'                      
                 
    """ 
    
    if packing == 'hex':
        numCores = numCores * (math.sqrt(3) / 2)
    
    radius = math.sqrt(numCores) * spacing / 2
    if shape == 'circle':
        radius = radius * 2 / math.sqrt(math.pi)
       
    return radius   


def predict_spacing(radius = 100, num_cores = 30000, shape = 'circle', packing = 'hex'):
    """ Returns predicted core spacing required to produce a simulated bundle
    with the specified number of cores and specified radius.
    
    Keyword Arguments
        radius   : float
                   bundle radius, radius will be returned in same units (default = 100)            
        num_cores: int
                   number of cores (default = 30000)
        shape    : str
                   bundle shape, 'circle' (default) or 'square'          
        packing  : str
                   core packing, 'hex' (default) or 'square'   

    Returns:
        float    : centre-centre core spacing, in same units as radius


    """ 
    
    if packing == 'hex':
        numCores = num_cores * (math.sqrt(3) / 2)
        
    spacing = radius * 2 / math.sqrt(num_cores)
    
    if shape == 'circle':
        spacing = spacing / 2 * math.sqrt(math.pi)
       
    return spacing


def gaussian_2d(x, y, sigma):
    """ Returns value of 2D Gaussian function with standard deviation sigma 
    at position (x,y) 
    """   
    
    return np.exp( - ( x**2 + y**2) / (2 * sigma**2))
        


def core_centres(
        spacing = 1,
        radius = 100,
        packing = 'hex',
        shape='circle',
        position_sd = 0,
        mode = 'lattice',
        disp_sd = 0,
        corr_length = None,
    stack_radius_sd = 0.15,
    stack_corr_length = None,
    stack_drift_gain = 1.0,
    stack_passes = 2,
        seed = None,
        rng = None):
    """Returns x and y coordinates of simulated fibre core centres.

    Keyword Arguments
        spacing    : float
                     centre-centre core spacing (default = 1)
        radius     : float
                     bundle radius, same units as spacing (default = 100)
        packing    : str
                     core packing, 'hex' (default) or 'square'
        shape      : str
                     bundle shape, 'circle' (default) or 'square'
        position_sd: float
                     standard deviation of local (uncorrelated) jitter
        mode       : str
                     'lattice' (default), 'displaced_hex', or 'stacked_hex'
        disp_sd    : float
                     standard deviation of smooth correlated displacement field
        corr_length: float
                     correlation length for smooth displacement field. If None,
                     defaults to 10 * spacing.
        stack_radius_sd: float
                     standard deviation of exclusion-circle radius variation
                     for 'stacked_hex' mode, relative to spacing / 2.
        stack_corr_length: float
                     correlation length for spatial radius variation field in
                     'stacked_hex' mode. If None, defaults to 12 * spacing.
        stack_drift_gain: float
                     gain of overlap-driven shifts in 'stacked_hex' mode.
        stack_passes: int
                     overlap-resolution passes per core for 'stacked_hex'.
        seed       : int or None
                     random seed used when rng is not supplied
        rng        : np.random.Generator or None
                     random number generator to use
    """

    rng = _resolve_rng(seed=seed, rng=rng)
    coreX, coreY = regular_lattice_centres(spacing=spacing, radius=radius, packing=packing)

    if mode == 'displaced_hex':
        coreX, coreY = apply_correlated_shifts(
            coreX,
            coreY,
            spacing=spacing,
            disp_sd=disp_sd,
            corr_length=corr_length,
            rng=rng
        )
    elif mode == 'stacked_hex':
        coreX, coreY = apply_stacked_circle_drift(
            coreX,
            coreY,
            spacing=spacing,
            radius_sd=stack_radius_sd,
            radius_corr_length=stack_corr_length,
            drift_gain=stack_drift_gain,
            stack_passes=stack_passes,
            rng=rng
        )
    elif mode != 'lattice':
        raise ValueError("mode must be 'lattice', 'displaced_hex', or 'stacked_hex'.")

    coreX, coreY = apply_random_nudges(coreX, coreY, position_sd=position_sd, rng=rng)
    coreX, coreY = _clip_core_positions(coreX, coreY, radius=radius, shape=shape)

    return coreX, coreY



def core_nudge(coreX, coreY, mag, distWeight, targetCore, angle):
    
    
    dist = np.sqrt((coreX - coreX[targetCore]) **2 + (coreY - coreY[targetCore]) **2 )
    angle2 = np.arctan((coreX - coreX[targetCore]) /(coreY - coreY[targetCore]) )
    dist[targetCore] = distWeight
    angle2[targetCore] = 0
    
    coreX = coreX + np.sin(angle) * mag * (distWeight / dist)**(1/4)
    coreY = coreY + np.cos(angle) * mag * (distWeight / dist)**(1/4)
    
    return coreX, coreY
    

#def core_centres_stack(spacing = 1, radius = 100, packing = 'hex', shape='circle', positionSD = 0):
 #   num = int(radius/spacing) * 2 + 1
  

if __name__ == "__main__":
    
    
    pixelSize = .62
    gridSize = (200,200)
    bundle_centre_x = gridSize[0]/2
    bundle_centre_y = gridSize[1]/2
    sigma = .6
    sigmaSD = .05
    position_sd = 0
    radius = 50
    spacing = 3
    coreIntensity = 1
    coreIntensitySD = 0.1
    
    
    
    core_x, core_y = regular_lattice_centres(spacing, radius, packing = 'hex')
    
    plt.figure();
    plt.plot(core_x, core_y, '.')
    
    #core_x, core_y = apply_random_nudges(core_x, core_y, position_sd=.1, seed=None, rng=None)
    #plt.figure();
    #plt.plot(core_x, core_y, '.')
    
    
    #core_x, core_y = apply_correlated_shifts(core_x, core_y, spacing=spacing, disp_sd=2, corr_length=None, seed=None, rng=None)
    #plt.figure();
    #plt.plot(core_x, core_y, '.')

    # Quick smoke test for the stacked-core mode.
    #base_x, base_y = regular_lattice_centres(spacing, radius, packing='hex')
   # stacked_x, stacked_y = apply_stacked_circle_drift(
    #    base_x,
     #   base_y,
      #  spacing=spacing,
       # radius_sd=0.25,
       # radius_corr_length=12 * spacing,
       # drift_gain=1.0,
       # stack_passes=2,
       # seed=3
    #)

    #mean_disp = np.mean(np.sqrt((stacked_x - base_x) ** 2 + (stacked_y - base_y) ** 2))
    #print(f"stacked_hex smoke test -> n_cores={len(stacked_x)}, mean displacement={mean_disp:.3f}")

    #plt.figure();
    #plt.plot(base_x, base_y, '.', alpha=0.35, label='base lattice')
    #plt.plot(stacked_x, stacked_y, '.', alpha=0.75, label='stacked drift')
    #plt.axis('equal')
    #plt.legend()
    #plt.title('Stacked-core drift test')

    # Smoke test for Voronoi-grain + lattice + relaxation generator.
    hybrid_positions = generate_fused_bundle_core_positions(
        domain_size=(2 * radius, 2 * radius),
        core_spacing=spacing,
        n_grains=14,
        relax_steps=1800,
        dt=0.002,
        noise_sigma=0.03 * spacing,
        seed=11
    )

    if hybrid_positions.shape[0] > 1:
        if cKDTree is not None:
            dists, _ = cKDTree(hybrid_positions).query(hybrid_positions, k=2)
            nn = dists[:, 1]
        else:
            diff = hybrid_positions[:, None, :] - hybrid_positions[None, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            nn = np.min(dist, axis=1)

        print(
            "hybrid smoke test -> "
            f"n_cores={hybrid_positions.shape[0]}, "
            f"mean nn={np.mean(nn):.3f}, min nn={np.min(nn):.3f}"
        )

    plt.figure();
    plt.plot(hybrid_positions[:, 0], hybrid_positions[:, 1], '.', markersize=2)
    plt.axis('equal')
    plt.title('Voronoi-grain hybrid core test')
    plt.show()
    
    
    # # Generate core positions
    # coreX, coreY = core_centres(spacing = spacing, radius = radius, position_sd = position_sd)
    # num_cores = np.shape(coreX)[0]
   
    
    # for i in range(30):
    #     n = np.random.randint(num_cores)
    #     ang = np.random.uniform(0, 2 * math.pi)
    #     coreX, coreY = core_nudge(coreX, coreY, 2, 8, n, ang)
    # target = 50
    # #print(coreX[target], coreY[target])
    
    # #coreX, coreY = core_nudge(coreX, coreY, 3, 3, target, 0)
    
    # #print(coreX[target], coreY[target])
    
        
    # numCores = np.shape(coreX)[0]
    
    # sigmas = np.random.normal(loc = sigma, scale = sigmaSD, size = numCores)
    # intensities = np.random.normal(loc = coreIntensity, scale = coreIntensitySD, size = numCores)
    
    
    # bundle = np.zeros(gridSize)
    
    # coreGridSize = math.ceil(sigma * 3)
    
    
    # #cores = core_functions_gaussian(np.shape(coreX)[0], sigmaMean = sigma / pixelSize, sigmaSD = sigmaSD / pixelSize)
    # nFails = 0
    # for cx, cy, coreSigma, coreIntensity in zip(coreX, coreY, sigmas, intensities):
        
    #     cx_pixel = bundle_centre_x + cx / pixelSize
    #     cy_pixel = bundle_centre_y + cy / pixelSize
    
    #     centrePixelX = round(cx_pixel)
    #     centrePixelY = round(cy_pixel)
        
    #     for ox in np.arange(centrePixelX - coreGridSize, centrePixelX + coreGridSize):
    #         for oy in np.arange(centrePixelY - coreGridSize, centrePixelY + coreGridSize):
    #             try:
    #                 bundle[oy, ox] = bundle[oy, ox] +  coreIntensity * gaussian_2d(oy - cy_pixel, ox - cx_pixel, coreSigma / pixelSize)   
    #             except:
    #                 nFails+=1
        
      
            
    # print(nFails/len(coreX))        
            
            
        
    
    # plt.figure(dpi = 600)
    # #plt.plot(coreX / pixelSize + 250, coreY / pixelSize + 250, 'x')
    # plt.imshow(bundle, cmap='gray', interpolation = 'nearest')
    # #plt.plot(100 + coreX[target]/ pixelSize, 100 + coreY[target]/pixelSize,'x')
    # im = Image.fromarray(bundle)
    # im.save('bundle.tif')
    
    
        
    
    

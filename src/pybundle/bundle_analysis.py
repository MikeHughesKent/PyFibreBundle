# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This module contains analysis functions for fibre bundle core geometry.
"""

import numpy as np
from scipy.spatial import cKDTree, Delaunay


def _triangle_circumradius(points):
    """Return circumradius of a triangle defined by three 2D points.

    Arguments:

    points : np.ndarray
            Array of shape (3, 2) containing the (x, y) coordinates of the three
            triangle vertices.

    Returns:
    
    float : Circumradius of the triangle. If the points are collinear, returns
        np.inf.

    """
    side_a = np.linalg.norm(points[1] - points[0])
    side_b = np.linalg.norm(points[2] - points[1])
    side_c = np.linalg.norm(points[2] - points[0])
    semiperimeter = 0.5 * (side_a + side_b + side_c)
    area_sq = semiperimeter * (semiperimeter - side_a) * (semiperimeter - side_b) * (semiperimeter - side_c)

    if area_sq <= 0:
        return np.inf

    area = np.sqrt(area_sq)
    return side_a * side_b * side_c / (4.0 * area)


def _concave_hull_vertex_indices(points, alpha_radius):
    """Return indices of points that lie on the concave hull.

    The hull is constructed with an alpha-shape procedure. In practice this
    gives a concave or "complex" hull that follows bundle notches and local
    irregularities more closely than a convex hull. The returned indices are
    the vertices that lie on that hull.

    Arguments:
        points: np.ndarray
                Array of shape (N, 2) containing the (x, y) coordinates of the points.
        alpha_radius :  float
                        Alpha-shape scale parameter expressed in the same units as the point
                        coordinates. Larger values move the hull towards a convex hull, while
                        smaller values allow the hull to follow smaller-scale notches.
    Returns:
        np.ndarray:  1D array containing the indices of the points that lie on
        the concave hull. The returned indices are sorted in ascending order.
    """
    tri = Delaunay(points)
    kept_simplices = []

    for simplex in tri.simplices:
        triangle_points = points[simplex]
        if _triangle_circumradius(triangle_points) <= alpha_radius:
            kept_simplices.append(simplex)

    if not kept_simplices:
        return np.unique(tri.convex_hull.reshape(-1))

    edge_counts = {}
    for simplex in kept_simplices:
        simplex = [int(simplex[0]), int(simplex[1]), int(simplex[2])]
        for edge in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[0], simplex[2])):
            edge = tuple(sorted(edge))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    hull_edges = [edge for edge, count in edge_counts.items() if count == 1]
    if not hull_edges:
        return np.unique(tri.convex_hull.reshape(-1))

    return np.unique(np.asarray(hull_edges).reshape(-1))


def _expand_boundary_indices(boundary_idx, tri, depth):
    """Expand a boundary set through Delaunay neighbours for a given number of layers.
    This is used to when wanting to exclude not just the hull points from an analysis
    but also one or more adjacent inner rings of points.
    
    Arguments:
        boundary_idx    : np.ndarray
                          1D array of integer indices corresponding to the initial boundary set
        tri :             scipy.spatial.Delaunay 
                          Delaunay triangulation of the full point set. This is used to find
                          neighbours of the boundary points. 
        depth :           int
                          Number of layers to expand. A value of 1 means only the initial boundary
                          points are returned, while a value of 2 also includes all Delaunay    
                          neighbours of those points, and so on. Must be at least 1.
    Returns:
        np.ndarray : 1D array of integer indices corresponding to the expanded boundary set. The
        returned indices are sorted in ascending order.
    """ 
    if depth <= 1:
        return boundary_idx

    adjacency = [set() for _ in range(np.max(tri.simplices) + 1)]
    for simplex in tri.simplices:
        a, b, c = [int(v) for v in simplex]
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    expanded = set(int(v) for v in boundary_idx)
    frontier = set(expanded)

    for _ in range(depth - 1):
        next_frontier = set()
        for vertex in frontier:
            next_frontier.update(adjacency[vertex])
        next_frontier.difference_update(expanded)
        expanded.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    return np.array(sorted(expanded), dtype=int)


def core_pattern_statistics(
        coreX,
        coreY,
        k_neighbours=6,
        exclude_boundary=True,
        boundary_alpha=None,
        boundary_layers=1):
    """Compute neighbour statistics for core positions.

    This function quantifies how close a core arrangement is to an ideal
    hexagonal lattice while also reporting large-scale disorder indicators.
    
    It uses two complementary neighbourhood models:

    1) k-nearest neighbours (KD-tree):
       Used for spacing and local bond-order metrics.
       - For each core i, the nearest-neighbour distance is d_i (the distance
         to its single closest non-self neighbour).
       - Summary spacing metrics are then:
         * mean_nearest_neighbour_dist = mean(d_i)
         * std_nearest_neighbour_dist  = std(d_i)
         * cv_nearest_neighbour_dist   = std(d_i) / mean(d_i)

       Local hexagonal order is measured via the complex bond-order parameter:

           psi6_i = (1/N_i) * sum_j exp(1j * 6 * theta_ij)

       where theta_ij is the angle from core i to neighbour j and N_i is the
       number of neighbours used (k_neighbours by default).
       The magnitude |psi6_i| is near 1 for locally hexagonal packing and
       smaller for disordered neighbourhoods.

    2) Delaunay triangulation graph:
       Used for coordination-number statistics (how many topological neighbours
       each core has). In an ideal hex lattice, most interior cores have
       coordination number 6. Deviations (e.g. 5 or 7) correspond to defects.

    Boundary handling
    -----------------
    Statistics can be dominated by bundle-edge effects because cores on the
    outer boundary do not have a full neighbourhood. For example, their local
    bond-order metric is biased low and their Delaunay coordination number is
    often less than 6 simply because there are no neighbours outside the image
    or outside the physical bundle.

    To reduce this bias, boundary cores are excluded by default. The exclusion
    set is obtained by first computing a concave hull of the point cloud and
    then removing the cores that lie on that hull:

    1) A Delaunay triangulation is built from all cores.
    2) An alpha-shape is used to extract a concave hull from the triangulation.
    3) Cores that lie on that concave hull are marked as boundary points.

    This follows local concavities better than a convex hull, which is useful
    for real bundles whose detected outline may be irregular. The parameter
    boundary_alpha controls the concave-hull scale in units of the median
    nearest-neighbour spacing. Larger values move the estimate towards a convex
    hull, while smaller values make the hull follow smaller-scale notches.

    Arguments:
    
        coreX : array-like
            X coordinates of core centres. Can be any shape; values are flattened.
        coreY : array-like
            Y coordinates of core centres. Must have same number of elements as
            coreX.
        k_neighbours : int, optional
            Number of nearest neighbours used for local bond-order analysis.
            Default is 6 (appropriate for hex-like packing).
        exclude_boundary : bool, optional
            If True, exclude cores on the estimated concave hull before computing
            summary statistics. Default is True.
        boundary_alpha : float or None, optional
            Concave-hull scale parameter expressed in units of the median nearest-
            neighbour spacing. Internally this is passed to the alpha-shape hull
            construction. If None, a value of 2.5 is used.
        boundary_layers : int, optional
            Number of Delaunay-neighbour layers to exclude starting from the
            concave hull. A value of 1 excludes only the hull points themselves.
            Larger values also remove one or more adjacent inner rings. Default
            is 1.

    Returns:
    
        dict
            Dictionary of scalar summaries and per-core arrays:
            - num_input_cores : int
            - num_cores : int, number of analysed cores after boundary exclusion
            - num_boundary_cores : int
            - analysis_mask : np.ndarray (N_input,) boolean mask of analysed cores
            - boundary_mask : np.ndarray (N_input,) boolean mask of excluded cores
            - nearest_neighbour_distances : np.ndarray (N_analysed,)
            - mean_nearest_neighbour_dist : float
            - std_nearest_neighbour_dist : float
            - cv_nearest_neighbour_dist : float
            - local_psi6 : np.ndarray (N_analysed,) complex values psi6_i
            - local_psi6_mage : np.ndarray (N_analysed,) equals |psi6_i|
            - mean_local_psi6_mag : float, mean(|psi6_i|)
            - global_psi6_mag : float, |mean(psi6_i)|
            - coord_num : np.ndarray (N_analysed,), Delaunay node degree
            - coord_hist : dict, keys are coordination numbers
            - defect_fraction_not_6 : float, fraction with coordination != 6
    """
    
    # Force core indices to be 1D arrays of the same length
    coreX = np.asarray(coreX).reshape(-1)
    coreY = np.asarray(coreY).reshape(-1)

    if coreX.size != coreY.size:
        raise ValueError("coreX and coreY must contain the same number of points.")

    num_input_cores = coreX.size
    if num_input_cores < 4:
        raise ValueError("At least 4 cores are required for neighbour analysis.")

    k_neighbours = int(k_neighbours)
    if k_neighbours < 1:
        raise ValueError("k_neighbours must be at least 1.")
    if boundary_layers < 1:
        raise ValueError("boundary_layers must be at least 1.")

    points = np.column_stack((coreX, coreY))

    # Compute nearest-neighbour distances for all cores using cKDTree for speed.
    tree = cKDTree(points)
    dists, neighbour_idx = tree.query(points, k=k_neighbours + 1)
    
    if dists.ndim == 1:  # Scipy implementation squeezes output to 1D if only 
                         # one neighbour is requested, ensure outputs are 2D arrays with shape (N, 2)
        dists = dists[:, np.newaxis]
        neighbour_idx = neighbour_idx[:, np.newaxis]
    
    
    all_nearest_neighbour_distances = dists[:, 1]
    median_spacing = float(np.median(all_nearest_neighbour_distances))

    analysis_mask = np.ones(num_input_cores, dtype=bool)
    boundary_mask = np.zeros(num_input_cores, dtype=bool)

    # Remove cores on edge of bundle to avoid biasing statistics. 
    if exclude_boundary:
        alpha_scale = 2.5 if boundary_alpha is None else float(boundary_alpha)
        alpha_radius = alpha_scale * median_spacing
        full_tri = Delaunay(points)
        boundary_idx = _concave_hull_vertex_indices(points, alpha_radius)
        boundary_idx = _expand_boundary_indices(boundary_idx, full_tri, depth=boundary_layers)
        boundary_mask[boundary_idx] = True
        analysis_mask = ~boundary_mask

    num_cores = int(np.sum(analysis_mask))

    if num_cores < 4:
        raise ValueError("Boundary exclusion leaves too few cores for neighbour analysis.")

    # Remove self-distance
    neighbour_points = points[neighbour_idx[:, 1:]]
    
    # Compute local bond-order parameter psi6 for all cores using the angles to their neighbours.
    centre_points = points[:, np.newaxis, :]
    delta = neighbour_points - centre_points
    theta = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    all_local_psi6 = np.mean(np.exp(1j * 6 * theta), axis=1)
    all_local_psi6_magnitude = np.abs(all_local_psi6)

    # Compute Delaunay triangulation and coordination number for all cores for defect statistics.
    tri = Delaunay(points)
    edges = set()
    for simplex in tri.simplices:
        a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((a, c))))

    all_coordination_number = np.zeros(num_input_cores, dtype=int)
    for i, j in edges:
        all_coordination_number[i] += 1
        all_coordination_number[j] += 1

    # Mask boundary cores and compute statistics when boundary cores are excluded.
    nearest_neighbour_distances = all_nearest_neighbour_distances[analysis_mask]
    local_psi6 = all_local_psi6[analysis_mask]
    local_psi6_magnitude = all_local_psi6_magnitude[analysis_mask]
    coordination_number = all_coordination_number[analysis_mask]

    mean_nn = float(np.mean(nearest_neighbour_distances))
    std_nn = float(np.std(nearest_neighbour_distances))
    cv_nn = float(std_nn / mean_nn) if mean_nn > 0 else np.nan
    mean_local_psi6_magnitude = float(np.mean(local_psi6_magnitude))
    global_psi6_magnitude = float(np.abs(np.mean(local_psi6)))

    unique_deg, counts_deg = np.unique(coordination_number, return_counts=True)
    coordination_histogram = {int(k): int(v) for k, v in zip(unique_deg, counts_deg)}
    defect_fraction_not_6 = float(np.mean(coordination_number != 6))

    return {
        'num_input_cores': int(num_input_cores),
        'num_cores': int(num_cores),
        'num_boundary_cores': int(np.sum(boundary_mask)),
        'analysis_mask': analysis_mask,
        'boundary_mask': boundary_mask,
        'nearest_neighbour_dists': nearest_neighbour_distances,
        'mean_nearest_neighbour_dist': mean_nn,
        'std_nearest_neighbour_dist': std_nn,
        'cv_nearest_neighbour_dist': cv_nn,
        'local_psi6': local_psi6,
        'local_psi6_mag': local_psi6_magnitude,
        'mean_local_psi6_mag': mean_local_psi6_magnitude,
        'global_psi6_mag': global_psi6_magnitude,
        'coord_num': coordination_number,
        'coord_hist': coordination_histogram,
        'defect_fraction': defect_fraction_not_6,
    }

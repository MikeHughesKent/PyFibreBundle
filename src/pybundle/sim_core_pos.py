# -*- coding: utf-8 -*-
"""
PyFibreBundle is an open source Python package for image processing of
fibre bundle images.

This module contains code for generating physically plausable fibre 
bundle core locations, usually for use in simulating fibre bundle images. 
"""

import time

import numpy as np
from scipy.spatial import cKDTree


class CorePacking:
    """Simulate core packing in a fibre bundle using a simple particle-based model. 
    Cores are represented as circles with random radii, and interact via a soft 
    repulsive force when they overlap. A boundary force keeps cores within a 
    circular region. The system is evolved using simple Euler integration, 
    and a grid-based spatial partitioning is used to efficiently compute 
    forces for large numbers of cores.
    
    Keyword Arguments:

    n_cores: int
        Number of cores to simulate (default: 30000)    
    radius_mean: float
        Mean core radius (default: 1.0)
    radius_std: float
        Standard deviation of core radius (default: 0.02)
    packing_fraction: float
        Desired packing fraction (default: 1.1)
    k: float
        Force constant for core-core repulsion (default: 1.0)
    k_boundary: float
        Force constant for boundary repulsion (default: 0.5)
    dt: float
        Time step for integration (default: 0.1)
    noise: float
        Standard deviation of random noise added to forces (default: 0.02, only applied in early steps)
    seed: int
        Random seed for reproducibility (default: 0)
    debug_timing: bool
        If True, print timing information for force computation and grid rebuilding (default: True)
    """

    def __init__(
        self,
        n_cores=30000,   
        bundle_radius=100,          
        radius_mean=1.0,
        radius_std=0.02,
        packing_fraction=1.1,
        k=1.0,
        k_boundary=0.5,
        dt=0.1,
        noise=0.02,
        seed=0,
        debug_timing=True,
    ):
        rng = np.random.default_rng(seed)

        self.n = n_cores
        self.bundle_radius = bundle_radius
        self.k = k
        self.kb = k_boundary
        self.dt = dt
        self.noise = noise
        self.rng = rng
        self.step_count = 0
        self.debug_timing = debug_timing
        self.radius_mean = radius_mean
        self.radius_std = radius_std 
        self.packing_fraction = packing_fraction


        # Generate core radii with random variation
        self.r = rng.normal(radius_mean, radius_std, n_cores)

        # Estimate bundle radius from packing 
        area_per_core =  np.mean(self.r) ** 2 / packing_fraction
        total_area = n_cores * area_per_core
        self.R = np.sqrt(total_area)

        # Generate random initial positions for cores
        theta = rng.uniform(0, 2*np.pi, n_cores)
        rad = self.R * np.sqrt(rng.uniform(0, 1, n_cores))
        self.x = np.stack([rad * np.cos(theta), rad * np.sin(theta)], axis=1)

        # --- grid setup ---
        self.cell_size = 2.5 * np.max(self.r)   # interaction range
        self._build_grid()

    # --------------------------------------------------
    # GRID (CELL LIST)
    # --------------------------------------------------

    def _build_grid(self):
        """Each core is assigned to a cell in a 2D grid. When computing pairwise
        interactions, only cores in the same or neighbouring cells are considered.
        """

        # Convert co-ordinates of each core to cell indices
        coords = (self.x + self.R) / self.cell_size
        self.cell_idx = coords.astype(int)

        # hash cells
        self.cell_dict = {}
        for i, c in enumerate(map(tuple, self.cell_idx)):
            self.cell_dict.setdefault(c, []).append(i)

        (
            self.cell_coords,
            self.cell_lookup,
            self.cell_offsets,
            self.cell_particles,
        ) = self._flatten_grid()

    def _neighbour_cells(self, cell):
        """Return neighbouring cell indices (including itself)."""
        cx, cy = cell
        return [
            (cx + dx, cy + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]

    # --------------------------------------------------
    # FORCE COMPUTATION
    # --------------------------------------------------
    def compute_forces(self, numba=True):

        if numba:
            F = compute_forces_numba(
                self.x,
                self.r,
                self.cell_coords,
                self.cell_lookup,
                self.cell_offsets,
                self.cell_particles,
                self.k,
                self.kb,
                self.R,
            )

            # noise (outside numba)
            if self.noise > 0 and self.step_count < 100:
                F += self.noise * self.rng.normal(size=F.shape)

            return F

        # Fall-back if not using Numba
        F = np.zeros_like(self.x)

        for cell, indices in self.cell_dict.items():
            # --- gather neighbour indices ---
            neighbour_cells = self._neighbour_cells(cell)

            neighbour_indices = []
            for nc in neighbour_cells:
                if nc in self.cell_dict:
                    neighbour_indices.extend(self.cell_dict[nc])

            if not neighbour_indices:
                continue

            j_idx = np.array(neighbour_indices)

            # --- cache neighbour data ONCE ---
            xj = self.x[j_idx]
            rj = self.r[j_idx]

            # --- loop over particles in this cell ---
            for i in indices:
                xi0 = self.x[i, 0]
                xi1 = self.x[i, 1]
                ri = self.r[i]

                # vector differences (no rij array)
                dx = xi0 - xj[:, 0]
                dy = xi1 - xj[:, 1]

                dist2 = dx*dx + dy*dy

                r_sum = ri + rj

                # valid interaction indices (no np.any)
                valid = np.where((dist2 < (r_sum * r_sum)) & (dist2 > 1e-12))[0]

                if valid.size == 0:
                    continue

                dx = dx[valid]
                dy = dy[valid]
                dist2_valid = dist2[valid]
                r_sum_valid = r_sum[valid]

                dist = np.sqrt(dist2_valid) + 1e-12
                overlap = r_sum_valid - dist

                force_mag = self.k * overlap

                # accumulate directly (no fij array)
                inv_dist = 1.0 / dist
                F[i, 0] += np.sum(force_mag * dx * inv_dist)
                F[i, 1] += np.sum(force_mag * dy * inv_dist)

        # --- boundary force ---
        norm = np.sqrt(np.sum(self.x**2, axis=1)) + 1e-12
        delta = norm + self.r - self.R
        mask = delta > 0

        F[mask] -= (
            self.kb * delta[mask][:, None] *
            (self.x[mask] / norm[mask][:, None])
        )

        # --- noise (only early) ---
        if self.noise > 0 and self.step_count < 100:
            F += self.noise * self.rng.normal(size=F.shape)

        return F
   
    def step(self):
        """Compute forces and update positions.
        """

        t1 = time.perf_counter() if self.debug_timing else 0.0
        F = self.compute_forces()
        if self.debug_timing:
            t2 = time.perf_counter()
            print(f"Step {self.step_count}: Force computation took {round(t2-t1,3)} s")

        self.x += self.dt * F
        if self.step_count % 10 == 0:
            t1 = time.perf_counter() if self.debug_timing else 0.0
            self._build_grid()
            if self.debug_timing:
                t2 = time.perf_counter()
                print(f"Step {self.step_count}: Grid rebuild took {round(t2-t1,3)} s")
        self.step_count += 1

 
    def run(self, n_steps=200, verbose=False):
        """Run the simulation for a given number of steps.
       
         Keyword Arguments:
            n_steps: int
                Number of steps to run (default: 200)
                verbose: bool   
                If True, print timing information for each step (default: False)
        Returns:
            tuple of pos, rad:
                pos: np.ndarray: Array of shape (n_cores, 2) with final core positions.
                rad: np.ndarray: Array of shape (n_cores,) with core radii.
        """
        for i in range(n_steps):
            self.step()
          
        return self.cores()

    def _flatten_grid(self):
        """Convert cell_dict into flat arrays for Numba.
        """
        
        cells = list(self.cell_dict.keys())
        n_cells = len(cells)

        cell_offsets = np.zeros(n_cells + 1, dtype=np.int64)
        cell_particles = []

        for i, cell in enumerate(cells):
            pts = self.cell_dict[cell]
            cell_particles.extend(pts)
            cell_offsets[i+1] = cell_offsets[i] + len(pts)

        cell_particles = np.array(cell_particles, dtype=np.int64)
        cell_coords = np.array(cells, dtype=np.int64)

        # Shift to non-negative coordinates and build O(1) lookup table for neighbour cells.
        minx = int(np.min(cell_coords[:, 0]))
        miny = int(np.min(cell_coords[:, 1]))
        cell_coords[:, 0] -= minx
        cell_coords[:, 1] -= miny

        nx = int(np.max(cell_coords[:, 0])) + 1
        ny = int(np.max(cell_coords[:, 1])) + 1
        cell_lookup = np.full((nx, ny), -1, dtype=np.int64)
        for idx in range(n_cells):
            cx = cell_coords[idx, 0]
            cy = cell_coords[idx, 1]
            cell_lookup[cx, cy] = idx

        return cell_coords, cell_lookup, cell_offsets, cell_particles

    def _nearest_core_distance_image(self, img_size=None):
        """Generate a 2D image where each pixel value is the distance to the nearest core.
        
        Points outside the bundle radius are set to zero.
        
        Keyword Arguments:
            img_size : int or None
                Size of the output image (img_size x img_size). If None, image size is
                chosen to sample the mean core spacing 5 times, i.e.,
                img_size = int(2 * R / (mean_spacing / 5))
        
        Returns:
            np.ndarray : 2D image of shape (img_size, img_size) with distances to nearest core.
        """
        # If no size specified, compute based on mean nearest-neighbor spacing
        if img_size is None:
            # Use all cores to estimate mean spacing
            tree = cKDTree(self.x)
            dists, _ = tree.query(self.x, k=2)  # k=2 to get 1st neighbor (excluding self)
            mean_spacing = np.mean(dists[:, 1]) if dists.ndim > 1 else np.mean(dists[1:])
            img_size = int(2 * self.R / (mean_spacing / 5))
            img_size = max(img_size, 64)  # Ensure reasonable minimum size
        
        # Create coordinate grid
        coord_range = np.linspace(-self.R, self.R, img_size)
        yy, xx = np.meshgrid(coord_range, coord_range)
        pixels = np.column_stack((xx.ravel(), yy.ravel()))
        
        # Compute distance to nearest core for each pixel
        tree = cKDTree(self.x)
        distances, _ = tree.query(pixels)
        
        # Reshape to image
        img = distances.reshape((img_size, img_size))
        
        # Mask pixels outside bundle radius
        rr = np.sqrt(xx**2 + yy**2)
        outside_bundle = rr > self.R
        img[outside_bundle] = 0.0
        
        return img, xx, yy

    def cores(self):
        """Return core positions and radii, scaled to give a bundle of the spacified
        radius."""
        scale = self.bundle_radius / np.max(np.sqrt(np.sum(self.x**2, axis=1)) + self.r)

        return self.x * scale, self.r * scale    


    def fill_gaps(self, num_add = 10):
        """Add new cores at positions of largest gaps in the current pattern, 
        as estimated by the nearest-core distance image. After this, run some
        steps to relax the pattern. This can help to speed up convergence for 
        large numbers of cores.

        Note that this changes the number of cores in the system, so should only 
        be used if you are not relying on a specific number of cores, or have 
        allowed for this by initially creating the system with a smaller 
        number of cores than desired.
        
        Keyword Arguments:
            num_add: int
                     Number of cores to add (default: 10)    
            
        """
        
        # Produce an image where brightest pixels are furthest from existing cores
        dist_img, xx, yy = self._nearest_core_distance_image()

        w,h = np.shape(dist_img)
        insert_coord = []

        # Run through, finding peaks and then masking area around peak before
        # we look for next one.
        area = self.radius_mean * 4
        for ii in range(num_add):
            peak = np.unravel_index(np.argmax(dist_img, axis=None), dist_img.shape)
            insert_coord.append((xx[peak[0], peak[1]], yy[peak[0], peak[1]]))
            dist_img[max(peak[0] - area,0): min(peak[0] + area,w), max(peak[1] - area,0): min(peak[1] + area,h)] = 0
            
        # Add cores
        self.x = np.vstack([self.x, insert_coord])
        self.r = np.hstack([self.r, np.full(num_add, self.radius_mean)])
        self.n += num_add


from numba import njit

@njit
def compute_forces_numba(x, r, cell_coords, cell_lookup, cell_offsets, cell_particles, 
                            k, kb, R):
        """ Numba-optimized version of compute_forces. The grid-based approach is implemented 
        in a way that is compatible with Numba, using flat arrays and explicit loops."""
        
        n = x.shape[0]
        F = np.zeros((n, 2))

        n_cells = cell_coords.shape[0]

        for c in range(n_cells):
            cx = cell_coords[c, 0]
            cy = cell_coords[c, 1]

            start_i = cell_offsets[c]
            end_i = cell_offsets[c+1]

            # loop over neighbouring cells
            for dcx in (-1, 0, 1):
                for dcy in (-1, 0, 1):

                    nx = cx + dcx
                    ny = cy + dcy

                    if nx < 0 or ny < 0 or nx >= cell_lookup.shape[0] or ny >= cell_lookup.shape[1]:
                        continue

                    nc = cell_lookup[nx, ny]
                    if nc < 0:
                        continue

                    # Process each cell-pair once.
                    if nc < c:
                        continue

                    start_j = cell_offsets[nc]
                    end_j = cell_offsets[nc+1]

                    if nc == c:
                        # Same cell: only compute upper triangle (ii < jj).
                        for ii in range(start_i, end_i):
                            i = cell_particles[ii]
                            xi0 = x[i, 0]
                            xi1 = x[i, 1]
                            ri = r[i]

                            for jj in range(ii + 1, end_i):
                                j = cell_particles[jj]

                                dx = xi0 - x[j, 0]
                                dy = xi1 - x[j, 1]

                                dist2 = dx*dx + dy*dy
                                if dist2 <= 1e-12:
                                    continue

                                rsum = ri + r[j]
                                if dist2 < rsum * rsum:
                                    dist = np.sqrt(dist2)
                                    overlap = rsum - dist
                                    f = k * overlap / (dist + 1e-12)

                                    fx = f * dx
                                    fy = f * dy

                                    F[i, 0] += fx
                                    F[i, 1] += fy
                                    F[j, 0] -= fx
                                    F[j, 1] -= fy
                    else:
                        # Different cells: compute all cross-pairs once.
                        for ii in range(start_i, end_i):
                            i = cell_particles[ii]
                            xi0 = x[i, 0]
                            xi1 = x[i, 1]
                            ri = r[i]

                            for jj in range(start_j, end_j):
                                j = cell_particles[jj]

                                dx = xi0 - x[j, 0]
                                dy = xi1 - x[j, 1]

                                dist2 = dx*dx + dy*dy
                                if dist2 <= 1e-12:
                                    continue

                                rsum = ri + r[j]
                                if dist2 < rsum * rsum:
                                    dist = np.sqrt(dist2)
                                    overlap = rsum - dist
                                    f = k * overlap / (dist + 1e-12)

                                    fx = f * dx
                                    fy = f * dy

                                    F[i, 0] += fx
                                    F[i, 1] += fy
                                    F[j, 0] -= fx
                                    F[j, 1] -= fy

        # --- boundary force ---
        for i in range(n):
            x0 = x[i, 0]
            x1 = x[i, 1]

            norm = np.sqrt(x0*x0 + x1*x1)
            delta = norm + r[i] - R

            if delta > 0:
                f = kb * delta / (norm + 1e-12)
                F[i, 0] -= f * x0
                F[i, 1] -= f * x1

        return F
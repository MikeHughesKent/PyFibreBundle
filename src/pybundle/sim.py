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

    Returns:
        float   : predicted number of cores in simulated bundle          
    """       

    if shape == 'circle':
        area = math.pi * radius**2
    elif shape == 'square':
        area = (2 * radius)**2
    else:
        raise ValueError("shape must be 'circle' or 'square'")

    if packing == 'hex':
        cell_factor = math.sqrt(3) / 2
    elif packing == 'square':
        cell_factor = 1.0
    else:
        raise ValueError("packing must be 'hex' or 'square'")

    # N = A_bundle / (cell_factor * d^2)
    num_cores = area / (cell_factor * spacing**2)

    return num_cores


def predict_bundle_radius(spacing = 1, num_cores = 30000, shape = 'circle', packing = 'hex'):
    """ Returns predicted radius required to produce a simulated bundle
    with the specified number of cores.
    
    Keyword Arguments
        spacing : float
                  centre-centre core spacing (default = 1)                  
        num_cores: int
                  number of cores (default = 30000)
        shape   : str
                  bundle shape, 'circle' (default) or 'square
        packing : str
                  core packing, 'hex' (default) or 'square' 

    Returns:
        float   : radius required to produce specified number of cores, in same units as spacing    
    """ 
    
    # --- packing geometry ---
    if packing == 'hex':
        cell_factor = math.sqrt(3) / 2   # area per core = factor * d^2
    elif packing == 'square':
        cell_factor = 1.0
    else:
        raise ValueError("packing must be 'hex' or 'square'")

    # total bundle area
    area = num_cores * cell_factor * spacing**2

    # --- convert area to radius ---
    if shape == 'circle':
        radius = math.sqrt(area / math.pi)
    elif shape == 'square':
        radius = math.sqrt(area) / 2
    else:
        raise ValueError("shape must be 'circle' or 'square'")

    return radius  


def predict_core_spacing(radius = 100, num_cores = 30000, shape = 'circle', packing = 'hex'):
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
    
    if shape == 'circle':
        area = math.pi * radius**2
    elif shape == 'square':
        area = (2 * radius)**2
    else:
        raise ValueError("shape must be 'circle' or 'square'")

    if packing == 'hex':
        cell_factor = math.sqrt(3) / 2
    elif packing == 'square':
        cell_factor = 1.0
    else:
        raise ValueError("packing must be 'hex' or 'square'")

    # A_bundle = N * cell_factor * d^2
    spacing = math.sqrt(area / (num_cores * cell_factor))

    return spacing


def regular_core_lattice(spacing=1, radius=100, packing='hex', shape = 'circle'):
    """Generate regular square or hex lattice core centres in either a circular or 
    square bundle shape. Packing can be a regular hexagonal or square lattice.
    
    Keyword Arguments:
        spacing : float
                  centre-centre core spacing (default = 1)                  
        radius  : float
                  bundle radius, same units as spacing (default = 100)
        packing : str
                  core packing, 'hex' (default) or 'square'
        shape   : str
                  bundle shape, 'circle' (default) or 'square'

    Returns:    
        coreX, coreY: 1D arrays of x and y coordinates of core centres, in same units as spacing     
    """
    
    assert packing in ['hex', 'square'], "packing must be 'hex' or 'square'"
    assert shape in ['circle', 'square'], "shape must be 'circle' or 'square'"
    
    numX = int(radius / spacing) * 2 

    ## If hex packing, need to generate more rows to get the same number of 
    # cores in y direction, because rows are staggered. 
    if packing == 'hex':
        numY = round(numX * 2 / math.sqrt(3))
    else:
        numY = numX

    coreX, coreY = np.meshgrid(
        np.linspace(-radius, radius, numX),
        np.linspace(-radius, radius, numY)
    )

    # For hex packing, shift every other row by half a spacing
    if packing == 'hex':
        coreX[1::2] = coreX[1::2] + float(spacing) / 2

    # Mask out cores outside the circle if shape is 'circle'
    if shape == 'circle':
        r = np.sqrt(coreX ** 2 + coreY ** 2)
        mask = r < radius
        coreX = coreX[mask]
        coreY = coreY[mask]

    coreX = np.squeeze(np.reshape(coreX, [coreX.size, 1]))
    coreY = np.squeeze(np.reshape(coreY, [coreY.size, 1]))

    return coreX, coreY

def _gaussian_2d(x, y, sigma, eccentricity = 1.0, angle = 0.0):
    """ Returns value of 2D Gaussian function with standard deviation sigma 
    at position (x,y). Optionally, an elliptical Gaussian can be generated 
    by specifying eccentricity and rotation angle in radians. 
    The Gaussian is normalised so that its integral over the whole plane is 1.

    """ 
    # Rotate coordinates by angle
    x_rot = x * np.cos(angle) + y * np.sin(angle)
    y_rot = -x * np.sin(angle) + y * np.cos(angle)
    
    # Apply eccentricity to y coordinate
    y_rot = y_rot * eccentricity    
    
    # Calculate Gaussian value
    g= np.exp( - ( x_rot**2 + y_rot**2) / (2 * sigma**2)) / (2 * np.pi * sigma**2 / eccentricity)  
    
    # Normalise so that integral over whole plane is 1
    return g / np.sum(g)

    #return np.exp( - ( x**2 + y**2) / (2 * sigma**2))
        

class BundleSim:
    """Class for simulating fibre bundle images. The core positions are generated 
    using the CorePacking class, and the image is generated by summing Gaussian 
    spots at each core location."""
    
    def __init__(self, 
                 core_x = None, 
                 core_y = None, 
                 core_radius = None, 
                 img_size = (512, 512), 
                 bundle_offset = (0,0),
                 pixel_size = 1.0):
        """Initialise BundleSim object with specified parameters.
        
        Arguments:
            coreX : 1D array
                    x coordinates of core centres
            coreY : 1D array
                    y coordinates of core centres
            core_radius : float
                         radius of cores in same units as coreX and coreY
            img_size : (int, int)
            pixel_size: float
                        size of pixels in same units as coreX and coreY
        """
        
        self.core_x = core_x
        self.core_y = core_y
        self.core_radius = core_radius
        self.img_size = img_size
        self.pixel_size = pixel_size
        self.bundle_offset = bundle_offset
        self.generate_eccentricity(0)


    def ref_image(self):
        """Generate simulated fibre bundle image by summing Gaussian spots at each 
        core location."""
        
        img = np.zeros(self.img_size)

        # Convert core coordinates to pixel coordinates, with bundle offset and centering
        core_px = self.core_x / self.pixel_size + self.img_size[0] / 2 + self.bundle_offset[0] / self.pixel_size
        core_py = self.core_y / self.pixel_size + self.img_size[1] / 2 + self.bundle_offset[1] / self.pixel_size
        core_pr = self.core_radius / self.pixel_size

        # Create a grid of pixel coordinates
        x = np.arange(self.img_size[1]) * self.pixel_size - self.img_size[1] * self.pixel_size / 2 + self.pixel_size / 2
        y = np.arange(self.img_size[0]) * self.pixel_size - self.img_size[0] * self.pixel_size / 2 + self.pixel_size / 2


        # Sum Gaussian spots for each core
        g_size = np.round(np.mean(self.core_radius / self.pixel_size) * 6)  # Size of grid to generate Gaussian spot on
        
        # Make g_size odd so that Gaussian is symmetric around core centre
        if g_size % 2 == 0:
            g_size += 1

        xv, yv = np.meshgrid(np.arange(g_size) - g_size // 2, np.arange(g_size) - g_size // 2)
        offset = int(g_size // 2)
        
        for cx, cy, cr, ecc, ecc_angle in zip(core_px, core_py, core_pr, self.eccentricity, self.core_rotation):

            sub_pixel_x = cx - int(cx) - 0.5
            sub_pixel_y = cy - int(cy) - 0.5
           
            insert = _gaussian_2d(yv - sub_pixel_y, xv - sub_pixel_x, cr, eccentricity = ecc, angle = ecc_angle)

            # Check if the Gaussian spot would be completely out of bounds, and if so, skip it
            if (cx + offset < 0) or (cx - offset >= self.img_size[1]) or (cy + offset < 0) or (cy - offset >= self.img_size[0]):
                continue

            # Check if the Gaussian spot would go partially out of bounds, and if so, trim it
            x_start = int(cx) - offset
            x_end = int(cx) + offset + 1    
            y_start = int(cy) - offset
            y_end = int(cy) + offset + 1

            if x_start < 0:
                insert = insert[:, -x_start:]
                x_start = 0
            if y_start < 0:
                insert = insert[-y_start:, :]
                y_start = 0
            if x_end > self.img_size[1]:
                insert = insert[:, :self.img_size[1] - x_end]
                x_end = self.img_size[1]
            if y_end > self.img_size[0]:
                insert = insert[:self.img_size[0] - y_end, :    ]
                y_end = self.img_size[0]


            img[y_start:y_end, x_start:x_end] += insert 

        return img
    

    def generate_eccentricity(self, eccentricity_sigma = 0.2):
        """Generate random eccentricity values for each core, drawn from a normal 
        distribution with mean 1 and specified standard deviation."""
        self.eccentricity =  np.random.normal(1.0, eccentricity_sigma, size=self.core_x.shape)
        self.core_rotation = np.random.uniform(0, 2 * np.pi, size=self.core_x.shape)



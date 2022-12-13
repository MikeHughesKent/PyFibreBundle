# PyFibreBundle
PyFibreBundle is a Python package for processing of images captured through optical fibre bundles. It is developed mainly by [Mike Hughes](https://research.kent.ac.uk/applied-optics/hughes) at the [Applied Optics Group](https://research.kent.ac.uk/applied-optics/), School of Physics and Astronomy, University of Kent. Bug reports, contributions and pull requests are welcome.

Full documentation is available [here](http://PyFibreBundle.readthedocs.io) and a summary of the current functionality is below:

The package was originally developed mostly for applications in endoscopic microscopy, including fluorescence endomicroscopy and holographic endomicroscopy, but there are also potential applications in endoscopy. The package is under active development, with one stable release (v1.0). 

The package is designed to be fast enough for use in imaging GUIs as well as for offline research - frame rates of over 100 fps can be achieved on mid-level hardware, including core removal and mosaicing. The Numba just-in-time compiler is used to accelerate key portions of code (particularly triangular linear interpolation) and OpenCV is used for fast mosaicing. If the Numba package is not installed then PyFibreBundle falls back on Python interpreted code.

The package is designed to be fast enough for use in imaging GUIs as well as for offline research - frame rates of over 100 fps can be achieved on mid-level hardware, including core removal and mosaicing. The Numba just-in-time compiler is used to accelerate key portions of code (particularly triangular linear interpolation) and OpenCV is used for fast mosaicing. If the Numba package is not installed then PyFibreBundle falls back on Python interpreted code.

## Capabilities

### Core Functions  
* Locate bundle in image.
* Crop image to only show bundle.
* Mask areas outside of bundle.
* Gaussian spatial filtering to remove core pattern.
* Determine core spacing.mdm
* Define and apply custom edge filter to remove core pattern.
* Find centers of all cores in bundle (two implementations: regional maxima and Hough transform).
* Core removal using triangular linear interpolation following Delaunay triangulation. 

### Mosaicing
* Detect image to image shift using normalised cross correlation.
* Insert image into mosaic either using dead-leaf or alpha blending.
* Expand or scroll mosaic when the edge of the mosaic image is reached.

### Super Resolution
* Combine multiple fibre bundle images to improve resolution.

Read the [full documentation](http://PyFibreBundle.readthedocs.io) for more details.

## Requirements

Required Packages:

* Numpy
* OpenCV
* Pillow
* Scipy

Optional Packages:

* Numba (for faster linear interpolation)
* Matplotlib (to run examples and tests)

## Acknowlegements
Cheng Yong Xin, Joseph, who contributed to triangular linear interpolation; Callum McCall who contributed to the super resolution component, Petros Giataganas who developed some of the Matlab code that parts of this library were ported from. 

Funding from EPSRC (Ultrathin fluorescence microscope in a needle, EP/R019274/1), Royal Society (Ultrathin Inline Holographic Microscopy) and University of Kent.

# PyFibreBundle
PyBundle is a Python library for image processing of images acquired through fibre imaging bundles. It is being developed mainly by [Mike Hughes](https://research.kent.ac.uk/applied-optics/hughes) at the [Applied Optics Group](https://research.kent.ac.uk/applied-optics/), University of Kent. Contributions and pull requests are welcome.

This library is currently under active development and there is no stable release. Note that some functions require the OpenCV library. 

Full documentation is available [here](http://PyFibreBundle.readthedocs.io) and a summary of the current functionality is below:

## Core Functions  
* Locate bundle in image
* Crop image to only show bundle
* Mask areas outside of bundle
* Gaussian spatial filtering to remove core pattern
* Determine core spacing
* Define and apply custom edge filter to remove core pattern
* Find centers of all cores in bundle (two implementations: regional maxima and Hough transform)
* Core removal using triangular linear interpolation following Delaunay triangulation. 

## Mosaicing
* Detect image to image shift using normalised cross correlation.
* Insert image into mosaic either using dead-leaf or alpha blending.
* Expand or scroll mosaic when the edge of the mosaic image is reached.

## Super Resolution
* Combine multiple fibre bundle images to improve resolution.

## PyBundle Class
* Object oriented version of PyFibreBundle

__Acknowledgements__: Cheng Yong Xin, Joseph, contributed to triangular linear interpolation; Petros Giataganas who developed some of the Matlab code that parts of this library were ported from. Funding from EPSRC (Ultrathin fluorescence microscope in a needle, EP/R019274/1), Royal Society (Ultrathin Inline Holographic Microscopy).

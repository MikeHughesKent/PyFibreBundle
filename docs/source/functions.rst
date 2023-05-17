------------------
Function Reference
------------------
A list of core functions is available below. Methods for the `Mosaic <mosaicing.html>`_ class are not listed here, please see the documentation pages for those classes separately.

The `PyBundle class <pybundle_class.html>`_ implements most of the functionality of the package and is the preferred approach for most applications except for Mosaicing, which is handled
by the `Mosaic class <mosaicing.html>`_. 

PyFibreBundle uses numpy arrays as images throughout, wherever 'image' is specified this refers to a 2D (monochrome) or 3D (colour) numpy array. For colour images, the colour channels are along the third axis. There can be as many colour channels as needed.


^^^^^^^^^^^^^^
Classes
^^^^^^^^^^^^^^

.. py:function:: PyBundle()

Provides object-oriented access to core functionality of PyFibreBundle. See `PyBundle class <pybundle_class.html>`_ for details.


.. py:function:: Mosaic()

Provides object-oriented access to mosaicing functionality of PyFibreBundle. See `Mosaic class <mosaicing.html>`_ for details.



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Low-Level Functions for Bundle finding, cropping, masking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: crop_rect(img, loc)

Crops a square image around bundle based on location specified by ``loc``, a tuple of ``(centre_x, centre_y, radius)``. Returns numpy array.


.. py:function:: find_bundle(img [,searchFilterSize = 4])

Finds the bundle in an image. Image is initially smoothed with a Gaussian filter of sigma ``searchFilterSize`` which should be of the order of, or larger than, the core spacing. Returns ``loc``, a tuple of ``(centre_x, centre_y, radius)``. 


.. py:function:: get_mask(img, loc)

Generates a mask image, 1 inside bundle and 0 outside of bundle, based on bundle location specified in ``loc``, a tuple of ``(centre_x, centre_y, radius)``. ``img`` can be any numpy array and merely defines the size of the mask. (i.e. ``mask`` will be the same size as ``img``). Returns numpy array.


.. py:function:: apply_mask(img, mask)

Applies a previously generated ``mask`` (e.g. from ``get_mask``) to an image ``img`` by multlying the two arrays. ``img`` and ``mask`` must be the same size. Returns numpy array.


.. py:function:: auto_mask(img, [,searchFilterSize])

Locates and masks an image ``img``. For meaning of ``searchFilterSize`` see ``find_bundle``. Returns numpy array.


.. py:function:: auto_mask_crop(img, [,searchFilterSize])

Locates, crops and masks an image ``img``. For meaning of ``searchFilterSize`` see ``find_bundle``. Returns numpy array.


.. py:function:: find_core_spacing(img)

Estimates the fibre core spacing in image ``img`` by looking for a peak in the power spectrum. Returns core spacing in pixels.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Low Level Functions for Spatial Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function::  g_filter(img, filterSize)

Applies a Gaussian filter to image ``img`` of sigma ``filterSize``. Returns numpy array.


.. py:function:: crop_filter_mask(img, loc, mask, filterSize, [,searchFilterSize])

Filters, crops and masks and image ``img`` using pre-defined mask ``mask`` and bundle location ``loc``, a 
a tuple of ``(centre_x, centre_y, radius)``. A Gaussian filter is applied of sigma ``filterSize``. For meaning of ``searchFilterSize`` see ``find_bundle``. Returns numpy array.


.. py:function:: edge_filter(imgSize, edgePos, edgeSlope)

Creates a Fourier domain filter for core removal based on a cosine smoothed edge filter at a spatial frequency corresponding to spatial distance ``edgePos``. The slope of the cut-off is given by ``edgeSlope``. Typical values are 1.6 and 0.1 times the core spacing, respectively. Returns numpy array.

.. py:function:: filter_image(img, filt)

Applies a Fourier domain filter ``filt`` (such as created by ``edge_filter``) to an image ``img``. Returns numpy array.


.. py:function:: smoothedImg = median_filter(img, filterSize)

Applies a median filter to image ``img`` of size ``filterSize`` which must be odd. Returns numpy array.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Functions for Triangular Linear Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
""""""""""""""""""""
High-level functions
""""""""""""""""""""

.. py:function::  calib_tri_interp(img, coreSize, gridSize[, centreX, centreY, radius, filterSize = 0,      normalise = None, autoMask = True, mask = True, background = None])

Calibration for triangular linear interpolation between cores. This returns a BundleCalibration, an object containig all the calibration information necessary for subsequent reconstructions.

Required arguments: 

* ``img`` calibraton image (2D/3D numpy array)
* ``coreSize`` estimate core spacing to help with core finding.
* ``gridSize`` size of output image (square)

*Optional arguments:*

* ``centreX``, ``centreY``, ``radius`` defines the area covered by the output image. If not specified, it will be centered on the bundle and include the full radius.
* ``filterSize`` sigma of Gaussian filter applied to images before extracting core intensities.
* ``normalise`` if a reference image is provided here, core intensities at reconstruction will be normalised with respect the core intensities in the reference image. This is generally necessary for good quality results.
* ``autoMask`` if ``true``, areas outside the bundle are set to 0 prior to locating cores. This generally helps to avoid spurious detections due to noise.
* ``mask`` if ``true``, a circular mask will be drawn around the bundle following reconstruction - this gives a less jagged edge to the image.
* ``background`` if a background image is provided here, this will be subtracted from image during the reconstruction stage.


.. py:function::  recon_tri_interp(img, calib, [useNumba = False])

Performs triangular linear interpolation on an image ``img`` using a calibration ``calib`` obtained from ``calib_tri_interp``. Set ``useNumba = True`` to use JIT compiler for speed-up (requires numba library to be installed). Returns a numpy array.

"""""""""""""""""""
Low-level functions
"""""""""""""""""""

.. py:function:: find_cores(img, coreSpacing)

A function used by ``calib_tri_interp`` to locate the bundle cores in the image ``img``. ``coreSpacing`` is the estimated core spacing in pixels which can be obtained using ``get_core_spacing`` if unknown. Returns tuple of ``(core_x, core_u)``, both 1D numpy arrays containing co-ordinates of each core.

.. py:function:: core_values(img, coreX, coreY, filterSize):

A function used by ``calib_tri_interp`` and ``recon_tri_interp`` to extract the intensity of each core in a image, based on core locations ``coreX`` and ``coreY``, which are 1D  numpy array, and ``filterSize`` which is the size of the Gaussian smoothing filter applied before extracting the intensities.

.. py:function:: init_tri_interp(img, coreX, coreY, centreX, centreY, radius, gridSize, **kwargs):

A function used by ``calib_tri_interp`` to perform Delaunay triangulation and to obtain the enclosing triangle for each reconstruction grid pixel.


^^^^^^^^^^^^^^^^^
Utility Functions
^^^^^^^^^^^^^^^^^

.. py:function:: extract_central(img, boxSize)

Extracts a central square from an image, of size ``boxSize``. Returns numpy array.

.. py:function:: to8bit(img [,minVal = None, maxVal = None]):

Converts an image to 8 bit. If ``minVal`` and ``maxVal`` are not specified, pixel values will be scaled so that everything lies in the range 0 to 255. Returns numpy array.

.. py:function:: radial_profile(img, centre)

Takes a radial profiles, averaged over all angles, from an image, centred on ``centre`` a tuple of ``(centre_x, centre_y)``. Returns 1D numpy array.
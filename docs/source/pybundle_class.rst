----------------------
PyBundle class
----------------------
The PyBundle class is the recommended way to use most functionality of the 
package (other than Mosaicing which has its own class. All method are listed
below, introductory guides on using the class are available for for 
`Basic Processing <core.rst>`_, `Linear Interpolation <linear_interp.html>`_
and `Super Resolution <super_res.html>`_ .

^^^^^^^^^^^^^^^
Instantiatation
^^^^^^^^^^^^^^^

.. py:function:: PyBundle(optional arguments)

Creates a PyBundle object. There are a large number of optional keyword
arguments (.e.g ``autoLoc = True``), which are listed below with their defaults
if not set. Each option also has a setter method (e.g. ``set_auto_loc``) which
can be called after creating the object as an alternative to passing the 
keyword arguement at creation. See the documentation for each setter, below,
for a detailed description of each option's meaning.

GENERAL Settings:

* autoContrast = False (``set_auto_contrast``)
* background = None  (``set_background``)
* coreMethod = None (``set_core_method``)

* outputType = Float (``set_output_type``)

CROP/MASK Settings (for FILTER/EDGE_FILTER only):

* applyMask = False (``set_apply_mask``)
* autoMask = True (``set_auto_mask``)
* autoLoc = False (``set_auto_loc``)
* crop = False(``set_crop``)
* loc = None (``set_loc``)
* mask = None (``set_mask``)


BACKGROUND Settings:

* backgroundImage = None (``set_background``)

FILTER Settings:

* filterSize = None (``set_filter_size``)

EDGE_FILTER Settings:

* edgeFilterShape = None (``set_edge_filter_shape``)

TRILIN Settings:

* calibImage = None (``set_calib_image``)
* normaliseImage = none (``set_normalise_image``)
* coreSize = 3 (``set_core_size``)
* autoMask = True (``set_auto_mask``)
* gridSize  = 512 (``set_grid_size``)
* useNumba = True (``set_use_numba``)
    
SUPER RESOLUTION Settings: 
   
* superRes = False (``set_super_res``)
* srShifts = None (``set_sr_shifts``)
* srCalibImages = None (``set_sr_calib_images``)
* srNormToBackgrounds = False (``set_sr_norm_to_backgrounds``)
* srNormToImages = True (``set_sr_norm_to_images``)
* srMultiBackgrounds = False (``set_sr_multi_backgrounds``)
* srMultiNormalisation = False (``set_sr_multi_normalisation``)
* srDarkFrame = None (``set_sr_dark_frame``)
* srUseLut = False (``set_sr_use_lut``)
* srParamValue = None (``set_sr_param_value``)
 
 
^^^^^^^^^^^^^^^
Main Methods
^^^^^^^^^^^^^^^

.. py:function:: process(img)

Process a raw image ``img`` which should be a 2D/3D numpy array using the 
current options. Returns a processed image as a 2D/3D numpy array.

.. py:function:: calibrate()

Performs the prior calibration necessary for ``TRILIN`` method and optional
for ``FILTER`` and ``EDGE_FILTER``. Most processing options (setters) should be 
called prior to this, although if ``set_background`` and ``set_normalise_image`` 
can be called later.


.. py:function:: get_pixel_scale()

Returns the scaling factor between pixels in the raw images and pixels in the processed images. 
This is always 1 for `FILTER` and `EDGE_FILTER` methods. For `TRILIN` method this will only return a valid 
value once ``calibrate()`` has been called, otherwise it will return ``None``.



"""""""""""""""
Setter Methods
"""""""""""""""

.. py:function:: set_auto_loc(bool)

If ``bool`` is ``True`` then the location of the bundle will be determined
automically for cropping and masking for ``FILTER`` or ``EDGE`` methods. This
has no effect on `TRILIN``. Default is ``True``.


.. py:function:: set_auto_contrast(bool)

If ``bool`` is ``True`` then the processed image is scaled to use the full dynamic 
range of the specified ``outputType``.  Default is ``False``.


.. py:function:: set_auto_mask(bool)

If ``bool`` is ``True`` then a mask will automaticall be created either from
the calibration image, if set, or otherwise the image to be processed. This will 
then be used for masking for ``FILTER`` or ``EDGE`` methods if ``crop`` is ``True``. This
has no effect on `TRILIN``. Default is ``True``.



.. py:function:: set_apply_mask(bool)

If ``bool`` is ``True``, images will be masked to set pixels outside of bundle 
to 0 when using ``FILTER`` or ``EDGE_FILTER`` methods. To generate this
automatically, the bundle location can 
be set using  ``set_loc``, otherwise is will be found automatically from the calibration 
image (if set) or the image to be processed. If ``set_auto_mask`` is set
``False`` and a mask is not provided no cropping will occur. Manually
provide a mask using ``set_mask()``.
 


.. py:function:: set_background(background)

Stores an image to be used for background subtraction. ``background`` should 
be a 2D/3D numpy array, the same size as the raw images to be processed. 
Pass ``None`` to remove the background image.


.. py:function:: set_bundle_loc(loc)

Sets the stored location of the fibre bundle. ``loc`` is a tuple of 
(centreX, centreY, radius).


.. py:function:: set_calib_image(calibImg)

Stores the image to be used for calibration method. ``calibImg`` should be a 
2D/3D numpy array of the same size as images to be processed, ideally showing 
the bundle with uniform illumination.


.. py:function:: set_core_method(coreMethod)

Sets which method will be used for core pattern removal, ``coreMethod`` can be 
``FILTER``, ``TRILIN`` or ``EDGE_FILTER``.


.. py:function:: set_core_size(coreSize)

Sets the estimated core spacing in the calibration image which helps with core 
finding as part of the TRILIN calibration process.


.. py:function:: set_crop(bool)

If ``bool`` is ``True``, images will be cropped to size of bundle when using 
``FILTER`` or ``EDGE_FILTER`` methods. The bundle location can be set using
 ``set_loc``, otherwise is will be found automatically from the calibration 
 image (if set) or the image to be processed. If ``set_auto_loc`` is set
 ``False`` and a bundle location is not provided, no cropping will occur.
 

.. py:function:: set_edge_filter_shape(edgePos, edgeSlope)

Sets the edge filter for use with EDGE_FILTER method. ``edgePos`` is the spatial 
frequency of the edge in pixels of FFT of image, ``edgeSlope`` is the 
steepness of slope (range from 10% to 90%) in pixels of the FFT of the image.


.. py:function:: set_filter_size(filterSize)

Sets the size of the Gaussian filter used by `FILTER` method in pixels.


.. py:function:: set_grid_size(gridSize)

Sets the size of the square output image for TRILIN method. ``gridsize`` 
should be an integer.


.. py:function:: set_mask(mask)

Sets the mask to be applied during processing to set areas outside bundle to 0. 
when ``set_apply_mask`` is ``True``. ``Mask`` is a 2D numpy array the same 
size as the raw images to be processed.


.. py:function:: set_normalise_image(normaliseImage)

Stores an image to be used for normalisation if TRILIN method is being used. 
``normaliseImage`` should be a 2D/3D numpy array, the same size as the raw 
images to be processed. Pass ``None`` to remove the normalisation image.


.. py:function:: set_output_type(outputType)

Set the data type of input images from 'process'. ``outputType`` should be one 
of ``'uint8'``, ``'unit16'`` or ``'float'``.


.. py:function:: set_use_numba(useNumba)

Determines whether Numba package is used for faster reconstruction for 
TRILIN method. ``useNumba`` is a booleab. Default is ``True``.


"""""""""""""""""""""""""""""""""""""""""""""
Super-Resolution Setter Methods
"""""""""""""""""""""""""""""""""""""""""""""

.. py:function:: set_super_res(superRes)

Enables super-resolution if ``superRes`` is ``True``, disables if ``False``.


.. py:function:: set_sr_calib_images(calibImages)

Provides the calibration images, a stack of shifted images used to determine 
shifts between images for super-resolution. ``calibImages`` is a 3D numpy 
array (x,y,nImages).
 
 
.. py:function:: set_sr_norm_to_images(normToImages)

Sets whether super-resolution recon should normalise each input image to have 
the same mean intensity. ``normToImages`` is Boolean.


.. py:function::  set_sr_norm_to_backgrounds(normToBackgrounds)

Sets whether super-resolution recon should normalise each input image with 
respect to a stack of backgrounds (provided using ``set_sr_backgrounds``) so 
as to have the same mean intensity. ``normToBackgrounds`` is Boolean.


.. py:function::  set_sr_multi_backgrounds(mb)

Sets whether super-resolution should perform background subtraction for each 
core in each image using a stack of background images (provided 
using ``set_sr_backgrounds``). ``mb`` is Boolean.

    
.. py:function:: set_sr_multi_normalisation(mn)

Sets whether super-resolution should normalise each core in each image using a 
stack of normalisation images (provided using ``set_sr_normalisation_images``). 
``mn`` is Boolean.
    
    
.. py:function:: set_sr_backgrounds(backgrounds)

Provide a set of background images for normalising intensity of each SR 
shifted image.


.. py:function:: set_sr_normalisation_images(normalisationImages)

Provide a set of normalisation images for normalising intensity of each SR 
shifted image.



.. py:function:: set_sr_shifts(shifts)

Provide known shifts between SR images instad of calculating them from a 
calibration stack. ``shifts`` is a 2D numpy array of (nImages,2). If set to 
``None`` (defualt) then the shifts are calculated from the calibration stack.


.. py:function:: set_sr_dark_frame(darkFrame)

Provide a dark background frame (i.e. with no optical power) which will be 
subtracted from each shifted super-resolution image.

.. py:function:: set_sr_use_lut(useLUT)

Enables or disables use of calibration LUT (if it has been created) for super 
resoution, ``useLUT`` is boolean.
    
.. py:function:: calibrate_sr_LUT(paramCalib, paramRange, nCalibrations) 

Creates a look up table (LUT) for TRILIN SR method. ``paramCalib`` is a 
calibration which maps the value of a parameter to the image shifts, as 
returned by ``calibrate_param_shifts``, ``paramRange`` is a tuple of 
(min, max) defining the range of values of the parameter to generate 
calibrations for, and ``nCalibrations`` if the number of calibrations to 
generate, equally spaced within this range.

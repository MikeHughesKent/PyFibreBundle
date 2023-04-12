----------------------
PyBundle class
----------------------
The PyBundle class is the recommended way to use most functionality of the package (other than Mosaicing which has its own class.)
The following gives a basic introduction to core removal using the class, to use the class for linear interpolation between cores, see the `Linear Interpolation Section <linear_interp.html>`_
and to use it for Super Resolution processing see the `Super Resolution Section <super_res.html>`_ 

Import the class using::

     from pybundle import PyBundle
    
and instantiate::

    pyb = PyBundle()
    
The processing parameters are then set, before calling::

    procImage = pyb.process(img)
    
to process an image ``img``, a 2D numpy array. This function returns a 2D numpy array as the processed image.

The processing parameters can be set using optional argument in the PyBundle constructor,
or using various setter methods. For example, if we want the processing to apply a Gaussian filter with a sigma of 2 pixels, we can use::

    pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2)
    
Alternatively, parameters can be set using the getter functions. The processing method is set using one of::

    pyb.set_core_method(pyb.FILTER)
    pyb.set_core_method(pyb.EDGE_FILTER)
    pyb.set_core_method(pyb.TRILIN)
    
FILTER removes cores by Gaussian filtering, EDGE_FILTER uses a custom smoothed edge filter created using the function ``edge_filter`` and TRILIN uses triangular linear interpolation.    
    
All methods support a background subtraction by specifying a background image, either by::

    pyb.set_background(backgroundImg)
  
or including ``backgroundImage = backgroundImg`` in the constructor, where ``backgroundImg`` is a 2D numpy array the same size as ``img``.        
    
Further examples of the settings relevant for filtering are in :doc:`Basic Processing<core>` and for TRILIN in :doc:`Linear Interpolation<linear_interp>`.

A full list of all functions of the PyBundle class is in :doc:`Function Reference<functions>`.

^^^^^^^^^^^^^^^^
Default Settings
^^^^^^^^^^^^^^^^
The default settings, and the associated setter methods are listed below. Each parameter
can be set in the constructor (e.g. ``pyb = Pybundle(gridSize = 20, autoContrast = True)`` or
subsequently using the setters. Settings should not be changed directly (e.g. ``pyb.gridSize = 100``) 
to avoid unexpected results and to maintain forwards compatibility. 
The meaning of each setting is explained in the :doc:`Function Reference<functions>`.

GENERAL Settings:

* autoContrast = False (``set_auto_contrast``)
* background = None  (``set_background``)
* coreMethod = None (``set_core_method``)
* loc = None (``set_bundle_loc``)
* mask = None (``set_mask``)
* outputType = 'uint16' (``set_output_type``)

BACKGROUND Settings:

* backgroundImage = None (``set_background``)

FILTER/EDGE_FILTER Settings:

* crop = False (``set_crop``)

FILTER Settings:

* filterSize = None (``set_filter_size``)

EDGE_FILTER Settings:

* edgeFilter = None (``set_edge_filter``)

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
 
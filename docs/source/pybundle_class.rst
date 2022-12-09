----------------------
PyBundle class
----------------------
The PyBundle class is the recommended way to use most functionality of the package (other than Super Resolution and Mosacing which have their own classes.)

Import the class using::

     from pybundle import PyBundle
    
and instantiate::

    pyb = PyBundle()
    
The processing parameters are then set, before calling::

    procImage = pyb.process(img)

to process an image ``img``, a 2D numpy array. This function returns a 2D numpy array as the processed image.

The processing method is set using one of::

    pyb.set_core_method(pyb.FILTER)
    pyb.set_core_method(pyb.EDGE_FILTER)
    pyb.set_core_method(pyb.TRILIN)
    
FILTER removes cores by Gaussian filtering, EDGE_FILTER uses a custom smoothed edge filter created using the function ``edge_filter`` and TRILIN uses triangular linear interpolation.    
    
All methods support a background subtraction by specifying a background image::

    pyb.set_background(backgroundImg)
  
where ``backgroundImg`` is a 2D numpy array the same size as ``img``.        
    
Further examples of the settings relevant for filtering are in :doc:`Basic Processing<core>` and for TRILIN in :doc:`Linear Interpolation<linear_interp>`.

A full list of all functions of the PyBundle class is in :doc:`Function Reference<functions>`.

^^^^^^^^^^^^^^^^
Default Settings
^^^^^^^^^^^^^^^^
The default settings, and the associated setter methods are listed below. Setters should be used rather than changing the settings directly to avoid unexpected results and forwards compatibility. The meaning of each setting is explained in the :doc:`Function Reference<functions>`.

General Settings:

* autoContrast = False (``set_auto_contrast``)
* background = None  (``set_background``)
* coreMethod = None (``set_core_method``)
* loc = None (``set_bunde_loc``)
* mask = None (``set_mask``)
* outputType = 'uint16' (``set_output_type``)

FILTER/EDGE_FILTER Settings:

* crop = False (``set_crop``)

FILTER Settings:

* filterSize = None (``set_filter_size``)

EDGE_FILTER Settings:

* edgeFilter = None (``set_edge_filter``)

TRILIN Settings:

* calibImage = None (``set_calib_image``)
* coreSize = 3 (``set_core_size``)
* doAutoMask = True (``set_auto_mask``)
* gridSize  = 512 (``set_grid_size``)
* useNumba = True  (``set_use_numba``)
    
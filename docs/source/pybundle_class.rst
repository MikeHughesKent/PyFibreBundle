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

**GENERAL Settings:**

* autoContrast = False (``set_auto_contrast``)
* background = None  (``set_background``)
* coreMethod = None (``set_core_method``)
* outputType = 'float64' (``set_output_type``)

**CROP/MASK Settings:**

* applyMask = False (``set_apply_mask``)
* autoMask = True (``set_auto_mask``)
* autoLoc = False (``set_auto_loc``)
* crop = False(``set_crop``)
* loc = None (``set_loc``)
* mask = None (``set_mask``)
* radius = None (``set_radius``)

**CALIB/BACKGROUND/NORMALISATION Settings:**

* calibImage = None (``set_calib_image``)
* backgroundImage = None (``set_background``)
* normaliseImage = none (``set_normalise_image``)

**GAUSSIAN FILTER Settings:**

* filterSize = None (``set_filter_size``)

**EDGE_FILTER Settings:**

* edgeFilterShape = None (``set_edge_filter_shape``)

**LINEAR INTERPOLATION Settings:**

* coreSize = 3 (``set_core_size``)
* gridSize  = 512 (``set_grid_size``)
* useNumba = True (``set_use_numba``)
    
**SUPER RESOLUTION Settings:**
   
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
Methods
^^^^^^^^^^^^^^^ 
 
.. autoclass:: pybundle.PyBundle
   :members:
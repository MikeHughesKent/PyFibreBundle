----------------------
Basic Image Processing
----------------------
PyFibreBundle includes functions for basic processing of bundle images, 
including locating, cropping and masking the bundle and removing the core 
pattern using spatial filtering or linear interpolation between cores. Both 
monochrome and colour images are supported. Monochrome images are stored
as 2D numpy arrays and colour images are stored as 3D numpy arrays, with the
colour channels along the third axis (any number of colour channels is allowed).
Image types explicitly supported are uint8, uint16, float32 and float64.

The recommended way to use this functionality is via the 
:doc:`PyBundle<pybundle_class>` class, but it is also possible to
use the `low level functions <low_level.html>`_ directly.

Full examples are available on Github for `spatial filtering <https://github.com/MikeHughesKent/PyFibreBundle/blob/main/examples/filtering_example.py>`_ 
and `linear interpolation <https://github.com/MikeHughesKent/PyFibreBundle/blob/main/examples/linear_interp_example.py>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the PyBundle class::

    from pybundle import PyBundle
    
We then instantiate a PyBundle object::

    pyb = PyBundle()
    
Let's assume we have an image stored in ``img``, a 2D (monochrome) or 3D 
(colour) numpy array. If the image is colour then the colour channels are along 
the third axis. 

In general, to process an image we use::

    procImage = pyb.process(img)

However, this will do nothing to the raw image unless we first set some processing options, either by
passing optional arguments when creating the PyBundle object, or by calling setter methods on the object
after creation.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
First we define what type of core-removal we would like, for example for Gaussian filtering with
a sigma of 2.5, we pass::

    pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2.5)
     
or equivalently::   

    pyb = PyBundle()
    pyb.set_core_method(PyBundle.FILTER)   # Choose to use a Gaussian filter
    pyb.set_filter_size(2.5)               # Gaussian filter sigma is 2.5 pixels
    
If we then call::

    procImage = pyb.process(img)

then ``procImage`` will be a Gaussian filtered version of ``img``.  

The other two options for core removal are PyBundle.TRILIN, for triangular linear
interpolation, and PyBundle.EDGE_FILTER to use a custom edge filter.

See the :doc:`Linear Interpolation<linear_interp>`  page for details on 
how to perform linear interpolation.

The edge filter is a spatial frequency domain filter that seeks to cut off
higher spatial frequencies includes those that correspond to the cores. It is used 
similarly to the Gaussian filter, except that the
filter size is defined by passing ``edgeFilterSize = (pos, slope)`` or
calling::

    set_edge_filter_size(pos, slope)
    
where ``pos`` the defines the position of the cut-off and ``slope`` defines
the steepness of the cut-off. ``pos`` should typically be around twice the 
core spacing and slope around 10% of this.   

As for the Gaussian filter, the best speed is achieved by setting a calibration
image and then calling ``calibrate()``; the edge filter will be generated at 
this point.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Background and Normalisation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can provide images which will be used for background subtraction and
normalisation (flat-fielding). Essentially, an image provided as background 
will be subtracted from the raw image to be processed, and the raw image
will be divided by the normalisation image (the exact implementation depends
on the core processing method).

If we have a background image in the numpy array ``backImg``, then to set the 
background image, pass ``background = backImg`` or call::

    pyb.set_background(backimg)

To set a normalisation image stored in ``normImg``, pass 
``normaliseImage = normImg`` or call::

    pyb.set_normalise_image(normImg)
    
Existing backgrounds/normalisations can be cleared by passing ``None`` to these
functions.        

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cropping 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we are using FILTER or EDGE_FILTER core removal methods, we might also want 
to crop the image to the size of the bundle (this happens intrinsically for
TRILIN). Cropping can be set to happen automatically by passing ``crop = True``
e.g. ::

    pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2.5, crop = True)

In this case, when ``process()`` is called, PyBundle will attempt to locate
the bundle and crop the image to a square just enclosing the bundle. 

This is inefficient when processing multiple images from the same imaging
setup, as the bundle location routine will run each time we call ``process()``.
It is faster to determine the location once from a calibration image, 
which is ideally an image with uniform illumination of the bundle
(although it could also be one of the images to be processed). 

Set the calibratation image by passing    
``calibImage = calibImg`` or by calling::

    pyb.set_calib_image(calibImg)
    
where ``calibImg`` is a 2D/3D numpy array containing the calibration image. The
calibration image should be the same size as the images to be processed.

Optionally, we can then call::

    pyb.calibrate()   
    
and PyBundle will calculate and store the location of the bundle. If 
``calibrate()`` is not called then the calibration image will still be used to
find the bundle for cropping when ``pyb.process()`` is called for the first time. The
bundle location will then be stored for future use.    

Alternatively, we can explicitly tell PyBundle the location of the bundle 
in advance, either by passing ``loc = location`` or by calling::

    pyb.set_loc(location)
    
where ``location`` is a tuple of (centre_x, centre_y, radius).



^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Masking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If we are using FILTER or EDGE_FILTER core removal methods, we might also want 
to set pixels outside the bundle to 0 (this happens intrinsically for
TRILIN). This masking can be set to happen automatically by passing 
``applyMask = True``, e.g. ::

    pyb = PyBundle(coreMethod = PyBundle.FILTER, filterSize = 2.5, applyMask = True)

As for cropping, PyBundle will generate a mask automatically each time we call 
pyb.calibrate() on an image which is generally not efficient. Again, it often better to 
generate the mask based on a calibration image in the same way as for cropping, i.e. by 
passing ``calibImage = calibImg``. Calling::

    pyb.calibrate()   

will then allow the mask to be generated in advance, otherwise it will
be created the first time we call pyb.process().


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Image Type and Autocontrast
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  
The default image output type is ``'float64'``, this can be changed by passing, for 
example ``outputType = 'uint8'`` when creating the ``PyBundle`` object, or by 
calling ::

    pyb.set_output_type('uint8')      # Output images will be 8 bit
    
where ``'uint8'``, ``'uint16'``, ``'float32'`` or ``'float64'`` can be used. 
The output will simply be cast to this format without any scaling, unless we pass 
``autoContrast = True`` or set::

   pyb.set_auto_contrast(True)     
  
in which case the image will be first scaled to between 0 and 255 if an 
8 bit output type is set, or between 0 and 65535 if a 16 bit output type is 
set, or between 0 and 1 if a floating point output type is set.

    


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Gaussian filtering: `examples/filtering_example.py <https://github.com/MikeHughesKent/PyFibreBundle/blob/main/examples/filtering_example.py>`_.

Linear interpolation: `examples/linear_interp_example.py <https://github.com/MikeHughesKent/PyFibreBundle/blob/main/examples/linear_interp_example.py>`_.

Colour Gaussian filtering: `examples/filtering_colour_example.py <https://github.com/MikeHughesKent/PyFibreBundle/blob/main/examples/filtering_colour_example.py>`_.

Colouor Linear interpolation: `examples/linear_interp_colour_example.py <https://github.com/MikeHughesKent/PyFibreBundle/blob/main/examples/linear_interp_colour_example.py>`_.
  
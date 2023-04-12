----------------------
Basic Image Processing
----------------------
PyFibreBundle includes several functions for basic processing of bundle images, including locating, cropping and masking the bundle and 
removing the core pattern using spatial filtering. The easiest way to use this functionality is via the :doc:`PyBundle<pybundle_class>` class, 
but it is also possible to use the lower-level functions directly.

^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started using OOP
^^^^^^^^^^^^^^^^^^^^^^^^^

Begin by importing the libary::

    from pybundle import PyBundle
    
Instantiate a PyBundle object::

    pyb = PyBundle()
    
To process an image ``img``, a 2D numpy array, we then use::

    procImage = pyb.process(img)

However, this will do nothing to the raw image unless we first set some parameters. Parameters can either be
set by passing optional arguments when instantiating the PyBundle object, or by calling setter methods. First we define what type of core-removal we would like, for example by passing arguments::

    pyb = PyBundle(coreMethod = pyb.FILTER, filterSize = 2.5)
     
or equivalently::   

    pyb = PyBundle()
    pyb.set_core_method(pyb.FILTER)   # Choose to use a Gaussian filter
    pyb.set_filter_size(2.5)          # Gaussian filter sigma is 2.5 pixels

We might also want to crop the image to the bundle. This can be done automically using::
    
    pyb = PyBundle(coreMethod = pyb.FILTER, filterSize = 2.5,
                   autoLoc = True, autoCrop = True)    

Or using the setters::
   
    pyb.set_auto_loc(calibImg)        # Automatically locate bundle in image
    pyb.set_crop(True)                # Output images will be cropped to a square around the bundle

where ``calibImg`` is a well-illuminated image which will allow the bundle to be located. This may be the same image as ``img``.

Alternatively, we can specify the location of the bundle by including ``bundleLoc = loc`` in the instantiation or::

    pyb.set_bundle_loc(loc)
    
where ``loc`` is a tuple of (xCentre, yCentre, radius) for the bundle.   

It is often useful to set all pixels outside the bundle to 0, which will be done if we pass ``autoMask = calibImg`` or ::

    pyb.set_auto_mask(calibImg)        

The output image type can be set by passing, for example ``outputType = 'uint8'``, or by calling ::

    pyb.set_output_type('uint8')      # Output images will be 8 bit
    
where ``'uint8'``, ``'uint16'`` or ``'float'`` can be used. The output will simply be cast to this format without any scaling, unless we pass ``autoContrast = True`` or set::

   pyb.set_auto_contrast(True)     
  
in which case the image will be first scaled to between 0 and 255 if an 8 bit output type is set, or between 0 and 65535 if a 16 bit output type is set, or between 0 and 1 if a floating point output type is set.


To use triangular linear interpolation rather than Gaussian filtering, pass ``coreMethod = PyBundle.TRILIN`` or set::

    pyb.set_core_method(pyb.TRILIN)
    
This requires a calibration - see the :doc:`Linear Interpolation<linear_interp>`  page for more details.  



""""""""
Examples
""""""""
An example of using the PyBundle class for filtering is in 'examples/filtering_example.py'.
An example of using the PyBundle class for linear interpolation is in 'examples/linear_interp_example.py'.

    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Getting Started Using Lower-Level Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The lower-level functions can be called directly if greater control is needed.

Begin by importing the library::
    
    import pybundle

To locate the bundle in an image, we use::

    loc = pybundle.find_bundle(img)

This works best for background or structureless images acquired through a bundle. The function returns ``loc`` which is a tuple of the bundles (x centre, y centre, radius).

If we would like to mask out pixels outside of the image, we can use::

    maskedImg = pybundle.auto_mask(img)

Alternatively, we can generate a mask using::

    mask = pybundle.get_mask(img, loc)

and apply this mask to any future image using::

    maskedImg = pybundle.apply_mask(img, mask)

This is more useful in general, since the location of the bundle is best determined using a calibration image, and the same mask can then be used for all subsequent images.

We can also crop the image to a square around the bundle using::

    croppedImg, newloc = pybundle.crop_rect(img, loc)

where we have specified the bundle location ``loc``, a tuple of (x centre, y centre, radius) as output by ``find_bundle``. Note that the output of is a tuple of ``(image, newloc)`` where ``newloc`` is the new location of the bundle in the cropped image.

To crop and mask an image in a single step use::

    croppedImg = pybundle.auto_mask_crop(img)

Spatial filtering can be used to remove the core pattern (alternatively, linear interpolation is also available). To apply a Gaussian smoothing filter, use::

    smoothedImg = pubundle.g_filter(img, filterSize)

where ``filterSize`` is the sigma of the 2D Gaussian smoothing kernel. A convenient function to filter, mask and crop an image is given by::

    smoothedImg = pybundle.crop_filter_mask(img, loc, mask)

where ``loc`` is the location of the bundle, determined using ``find_bundle`` on a calibraton image, and ``mask`` is a mask created by ``get_mask``.

The core spacing of the bundle can be found using::

    coreSpacing = pybundle.get_core_spacing(img)

This can then be used to define a custom edge filter using::

    filter = pybundle.edge_filter(img,  edgeLocation, edgeSlope)

This defines a Fourier domain filter with a cosine smoothed cut-off at the spatial frequency corresponding to the spatial distance ``edgeLocation``. ``edgeSlope`` defines the smoothness of the cut-off; a value of 0 gives a rectangular function. ``img`` merely needs to be a numpy array the same size as the image(s) to be filtered. ``edgeLocation`` should typically be ``1.6 * coreSpacing``, and ``edgeSlope`` is not critical, but a value of ``0.1 * coreSpacing`` generally works well. To apply the filter use::

    smoothedImg = pybundle.filter_image(img, filter)
   
Note that this kind of filtering is currently quite slow.    
    
To perform linear interpolation using the low-level functions, first perform a calibration using the calibration image ``calibImg``, a 2D numpy array::

    coreSize = 3
    gridSize = 512    
    calib = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, normalise = calibImg, automask = True)  

Here we have specified ``coreSize = 3`` which is the approximate core spacing in the image. This assists the calibration routine in finding all cores. If unknown it can be estimate using ``find_core_spacing``.

The ``gridSize`` is the number of pixels in each dimensions of the reconstructed image, which is square.

Finally, we have specified to use the ``calibImg`` for normalisation. This means that the intensity extracted from each core during imaging will be normalised with respect to the intensity from the calibration image, removing effects due to non-uniform cores. If this is not done (i.e. normalise is left as the default ``None``) then images may appear grainy.

To reconstruct an image ``img``, a 2D numpy array, we then call::

   imgRecon = pybundle.recon_tri_interp(img, calib)

This returns a 2D numpy array of size ``(gridSize, gridSize)`` containing the image with the core pattern removed.

For all optional parameters refer to the :doc:`function reference<functions>` for ``calib_tri_interp`` and ``recon_tri_interp``.


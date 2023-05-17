------------------------------
Using the Low-Level Functions
------------------------------
While the ``PyBundle`` class is the recommended way to use most functinality.
the lower-level functions can be called directly if greater control is needed.

The fuctions are documented fully in the `Function Reference <functions.html>`_.

Begin by importing the package::
    
    import pybundle

To locate the bundle in an image, we use::

    loc = pybundle.find_bundle(img)

This works best for background or structureless images acquired through a 
bundle. The function returns ``loc`` which is a tuple of the
(x_centre, y _entre, radius).

If we would like to mask out pixels outside of the image, we can use::

    maskedImg = pybundle.auto_mask(img)

Alternatively, we can generate a mask using::

    mask = pybundle.get_mask(img, loc)

and apply this mask to any future image using::

    maskedImg = pybundle.apply_mask(img, mask)

This is more useful in general, since the location of the bundle is best 
determined using a calibration image, and the same mask can then be used for 
all subsequent images.

We can also crop the image to a square around the bundle using::

    croppedImg, newloc = pybundle.crop_rect(img, loc)

where we have specified the bundle location ``loc``, a tuple of 
(x_centre, y_centre, radius) as output by ``find_bundle``. Note that the 
output is a tuple of ``(image, newloc)`` where ``newloc`` is the new location 
of the bundle in the cropped image.

To crop and mask an image in a single step use::

    croppedImg = pybundle.auto_mask_crop(img)

Spatial filtering can be used to remove the core pattern (alternatively, 
linear interpolation is also available). To apply a Gaussian smoothing filter, 
use::

    smoothedImg = pubundle.g_filter(img, filterSize)

where ``filterSize`` is the sigma of the 2D Gaussian smoothing kernel. A 
convenient function to filter, mask and crop an image is given by::

    smoothedImg = pybundle.crop_filter_mask(img, loc, mask)

where ``loc`` is the location of the bundle, determined using ``find_bundle`` 
on a calibraton image, and ``mask`` is a mask created by ``get_mask``.

The core spacing of the bundle can be found using::

    coreSpacing = pybundle.get_core_spacing(img)

This can then be used to define a custom edge filter using::

    filter = pybundle.edge_filter(img,  edgeLocation, edgeSlope)

This defines a Fourier domain filter with a cosine smoothed cut-off at the 
spatial frequency corresponding to the spatial distance ``edgeLocation``. 
``edgeSlope`` defines the smoothness of the cut-off; a value of 0 gives a 
rectangular function. ``img`` merely needs to be a numpy array the same size 
as the image(s) to be filtered. ``edgeLocation`` should typically be 
``1.6 * coreSpacing``, and ``edgeSlope`` is not critical, but a value of 
``0.1 * coreSpacing`` generally works well. To apply the filter use::

    smoothedImg = pybundle.filter_image(img, filter)
   
Note that this kind of filtering is currently quite slow.    
    
To perform linear interpolation using the low-level functions, first perform a 
calibration using the calibration image ``calibImg``, a 2D numpy array::

    coreSize = 3
    gridSize = 512    
    calib = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, 
                                     normalise = calibImg, automask = True)  

Here we have specified ``coreSize = 3`` which is the approximate core spacing 
in the image. This assists the calibration routine in finding all cores. If 
this is unknown it can be estimate using ``find_core_spacing()``.

The ``gridSize`` is the number of pixels in each dimensions of the 
reconstructed image, which is square.

Finally, we have specified to use the ``calibImg`` for normalisation. This 
means that the intensity extracted from each core during imaging will be 
normalised with respect to the intensity from the calibration image, 
removing effects due to non-uniform cores. If this is not done (i.e. normalise 
is left as the default ``None``) then images may appear grainy.

To reconstruct an image ``img``, a 2D numpy array, we then call::

   imgRecon = pybundle.recon_tri_interp(img, calib)

This returns a 2D/3D numpy array of size ``(gridSize, gridSize, colour channels)`` 
containing the image with the core pattern removed.

For all optional parameters refer to the :doc:`function reference<functions>` 
for ``calib_tri_interp`` and ``recon_tri_interp``.


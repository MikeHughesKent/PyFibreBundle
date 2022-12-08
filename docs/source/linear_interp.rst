Linear Interpolation
====================================
Triangular linear interpolation can be used to remove the fibre bundle core pattern. Using a calibration image, usually acquired with no object in view (i.e. a flat field), the location of each core is determined. A Delaunay triangulation is performed over the core locations. A reconstruction grid is then defined, and the enclosing triangle for each pixel is determined. Images can then be processed by interpolating the value of each pixel from the brightness of the three surrounding cores. Although calibration can take a few seconds, processing of images can then be at video rate.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Object Oriented Approach
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import the PyBundle class and instantiate a PyBundle object::

    from pybundle import PyBundle
    pyb = pybundle.PyBundle()
	
Set the core removal method to triangular linear interpolation::

    pyb.set_core_method(pyb.TRILIN)

Set both the calibration and normalisation images to be ``calibImg``, a 2D numpy array::

    pyb.set_calib_image(calibImg)
    pyb.set_normalise_image(calibImg)

Choose the output images size::

    pyb.set_grid_size(512)

If we are normalising it is best to get an output image which is auto-contrasted::

    pyb.set_auto_contrast(True)

Perform the calibration::

    pyb.calibrate()

Remove the fibre bundle pattern from an image ``img``, a 2D numpy array::

    imgProc = pyb.process(img)
    
For real-time processing, a speed-up of approx X4 in reconstruction can be obtained if the Numba package is installed. To disable use of Numba, call::

    pyb.set_use_numba(false)
    

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lower level functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For greater customisation, the static functions can be called directly. First perform a calibration using the calibration image ``calibImg``, a 2D numpy array::

    coreSize = 3
    gridSize = 512    
    calib = pybundle.calib_tri_interp(calibImg, coreSize, gridSize, normalise = calibImg, automask = True)  

Here we have specified ``coreSize = 3`` which is the approximate core spacing in the image. This assists the calibration routine in finding all cores. If unknown it can be estimate using ``find_core_spacing``.

The ``gridSize`` is the number of pixels in each dimensions of the reconstructed image, which is square.

Finally, we have specified to use the ``calibImg`` for normalisation. This means that the intensity extracted from each core during imaging will be normalised with respect to the intensity from the calibration image, removing effects due to non-uniform cores. If this is not done (i.e. normalise is left as the default ``None``) then images may appear grainy.

To reconstruct an image ``img``, a 2D numpy array, we then call::

   imgRecon = pybundle.recon_tri_interp(img, calib)

This returns a 2D numpy array of size ``(gridSize, gridSize)`` containing the image with the core pattern removed.

For all optional parameters refer to the functon reference for ``calib_tri_interp`` and ``recon_tri_interp``.


^^^^^^^
Example
^^^^^^^

An example using OOP is in "examples\\linear_interp_example.py".

An example using OOP and showing the difference in speed if Numba is used is in "examples\\linear_interp_numba_example.py".

Linear interpolation is compared with other core method removal techniques in "examples\\compare_recons.py"
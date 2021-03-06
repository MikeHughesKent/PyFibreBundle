Super Resolution
====================================
Core super-resolution (i.e. overcoming the sampling limit of the fibre bundle) can be achieved by combining multiple images, with the object slightly shifted with respect to the fibre pattern. The super-resolution class of PyFibreBundle provides the ability to combine multiple images and generate an enhanced resolution using triangular linear interpolation. This functionality is not currently accessible in the PyBundle class and must be invoked using static functons within the SuperRes class.

^^^^^^^^^^
Quickstart 
^^^^^^^^^^

Import the pybundle libary::

    import pybundle
   
First, perform the calibration. This requires a flat-field/background image ``calibImg``, a 2D numpy array, a stack of shifted images ``imgs``, a 3D numpy array (x,y,n), an estimate of the core spacing ``core size``, and the output image size ``gridSize`` ::

    calib = pybundle.SuperRes.calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, normalise = calibImg)

We have also specified an optional parameter, a normalisation image ``calibImg``, which prevents the images becoming grainy due to core-core variations. Note that ``imgs`` does not need to be the actual images to be used for reconstruction, but they must have the same relative shift as the the images. Alternatively, if the shifts are known, these can be specified using the optional parameter ``shifts`` which should be a 2D numpy array of the form (x_shift, y_shift, image_number).

We then perform the super-resolution reconstruction using::

    reconImg = SuperRes.recon_multi_tri_interp(imgs, calib)

which returns ``reconImg`` a 2D numpy array representing the output image.


^^^^^^^^^^^^^^^^^^^^^^
Implementation Details 
^^^^^^^^^^^^^^^^^^^^^^

``calib_multi_tri_interp`` first calls ``calib_tri_interp`` to perform the standard calibration for triangular linear interpolation. This obtains the core locations, using ``find_cores``. If the optional parameters ``normToImage`` or ``normToBackground`` are set to ``True``, then the mean image intensity for ether the images stack or the background stack (supplied as a further optional parameter ``backgroundImgs``) are calculated and stored. These are then later used to normalise each of the input images to a constant mean intensity. This is important for applications where the illumination intensity will be different for each image, but in most applications would not be needed.

``calib_multi_tri_interp`` then calculates the relative shifts between the supplied images in ``imgs`` using ``get_shifts`` via normalised cross correlation. Alternatively, shifts can be provided via the optional optional parameter ``shifts``. For each image, the recorded core positions are then translated by the measured shifts, and a single list of shifted core positions is assembled, containing the shifted core positions from all the images. ``init_tri_interp`` is then called, which forms a Delaunay triangulation over this set of core positions. For eacheach pixel in the reconstruction grid the enclosing triangle is identified and the pixel location in barycentric co-ordinates is recoreded.

Reconstruction is performed using ``recon_multi_tri_interp``.  The intensity value from each core in each of the images are extracted, and then pixel values in the final image are interpolated linearly from the three surrounding cores, using the pre-calculated barycentric distance weights.

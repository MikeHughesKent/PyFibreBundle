Super Resolution
====================================
Core super-resolution (i.e. overcoming the sampling limit of the fibre bundle) can be achieved by combining 
multiple images, with the object slightly shifted with respect to the fibre pattern. The super-resolution 
sub-package of PyFibreBundle provides the ability to combine multiple images and generate an enhanced resolution 
using triangular linear interpolation. This functionality is not currently accessible in the PyBundle class 
and must be invoked using functions wihtin the ``SuperRes`` class.

^^^^^^^^^^^^^^^^
Getting Started 
^^^^^^^^^^^^^^^^

Import the pybundle and pybundle super_res packages::

    import pybundle
    from pybundle import SuperRes
   
First, perform the calibration. This requires a flat-field/background image ``calibImg`` (a 2D numpy array), a stack of shifted images ``imgs`` (a 3D numpy array - [x,y,n]), an estimate of the core spacing ``core size``, and the output image size ``gridSize`` ::

    calib = pybundle.SuperRes.calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, normalise = calibImg)

We have also specified an optional parameter, a normalisation image ``calibImg``, which prevents the images becoming grainy due to core-core variations. Note that ``imgs`` does not need to be the actual images to be used for reconstruction, but they must have the same relative shift as the the images. Alternatively, if the shifts are known, these can be specified using the optional parameter ``shifts`` which should be a 2D numpy array of the form (x_shift, y_shift, image_number). If ``shifts`` is specified then ``imgs`` can be ``None``.

We then perform the super-resolution reconstruction using::

    reconImg = SuperRes.recon_multi_tri_interp(imgs, calib)

which returns ``reconImg`` a 2D numpy array representing the output image.


^^^^^^^^^^^^^^^^^^
Function Reference
^^^^^^^^^^^^^^^^^^

.. py:function:: calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, [optional arguments])

*Required arguments:*

* ``calibImg`` Calibration image of fibre bundle, 2D numpy array
* ``imgs`` Example set of images with the same set of mutual shifts as the images to later be used to recover an enhanced resolution image from. 3D numpy array. Can be ``None`` if ``shifts`` is specified instead.
* ``coreSize`` Estimate of average spacing between cores
* ``gridSize`` Output size of image, supply a single value, image will be square

*Optional arguments:*

* ``background`` Image used for background subtraction as 2D numpy array, defaults to no background
* ``normalise`` Image used for normalisation of core intensities, as 2D numpy array. Can be same as calibration image, defaults to no normalisation
* ``shifts`` Known x and y shifts between images as 2D numpy array of size (numImages,2). Will override ``imgs`` if specified as anything other than ``None``.
* ``centreX`` X centre location of bundle, if not specified will be determined automatically.
* ``centreY`` Y centre location of bundle, if not specified will be determined automatically.
* ``radius`` Radius of bundle, if not specified will be determined automatically.
* ``filterSize`` Sigma of Gaussian filter applied during core-finding, defaults to no filter.
* ``normToImage`` If ``true`` each image will be normalised to have the same mean intensity. Defaults to ``false``.
* ``normToBackground`` If ``true``, each image will be normalised with respect to the corresponding background image from a stack of background images (one for each shift position) provided in ``backgroundImgs``.
* ``backgroundImgs`` Stack of images, same size as ``imgs`` which are used to normalise
* ``imageScaleFactor`` If normToBackground and normToImage are False (default), use this to specify the normalisation factors for each image. Provide a 1D array the same size as the number of shifted images. Each image will be multiplied by the corresponding factor prior to reconstruction. Default is None (i.e. no scaling).
* ``postFilterSize`` Sigma of Gaussian filter applied to image after reconstruction, defaults to no filter.
* ``autoMask`` Whether to mask pixels outside bundle when searching for cores. Defualts to ``true``.
* ``mask`` Whether to mask pixels outside of bundle in reconstructed image. Defaults to ``true``.

*Returns:*

* Instance of ``BundleCalibration``

.. py:function:: recon_multi_tri_interp(imgs, calib, [optional arguments])

*Required arguments:*

* ``imgs`` Stack of shifted images as 3D numpy array. The third axis is image number.
* ``calib`` Instance of ``bundleCalibration`` returned by ``calib_multi_tri_interp``.

*Optional arguments:*

* ``useNumba`` Boolean, whether to use Numba package to speed up reconstruction if available. Defaults to ``true``.

*Returns:*

* Reconstructed image as 2D numpy array.

^^^^^^^^^^^^^^^^^^^^^^
Implementation Details 
^^^^^^^^^^^^^^^^^^^^^^

``calib_multi_tri_interp`` first calls ``calib_tri_interp`` to perform the standard calibration for triangular linear interpolation. This obtains the core locations, using ``find_cores``. If the optional parameters ``normToImage`` or ``normToBackground`` are set to ``True``, then the mean image intensity for ether the images stack or the background stack (supplied as a further optional parameter ``backgroundImgs``) are calculated and stored. These are then later used to normalise each of the input images to a constant mean intensity. This is important for applications where the illumination intensity will be different for each image, but in most applications would not be needed. It is also possible to provide a 1D array of normalisation factors directly as the ``imageScaleFactor`` parameter.

``calib_multi_tri_interp`` then calculates the relative shifts between the supplied images in ``imgs`` using ``get_shifts`` via normalised cross correlation. Alternatively, shifts can be provided via the optional parameter ``shifts``. For each image, the recorded core positions are then translated by the measured shifts, and a single list of shifted core positions is assembled, containing the shifted core positions from all the images. ``init_tri_interp`` is then called, which forms a Delaunay triangulation over this set of core positions. For each pixel in the reconstruction grid the enclosing triangle is identified and the pixel location in barycentric co-ordinates is recorded.

Reconstruction is performed using ``recon_multi_tri_interp``.  The intensity value from each core in each of the images are extracted, and then pixel values in the final image are interpolated linearly from the three surrounding (shifted) cores, using the pre-calculated barycentric distance weights.

^^^^^^^
Example
^^^^^^^

An example is provided in "examples\\super_res_example.py"
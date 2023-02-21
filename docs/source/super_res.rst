Super Resolution
====================================
Core super-resolution (i.e. overcoming the sampling limit of the fibre bundle) can be achieved by combining 
multiple images, with the object slightly shifted with respect to the fibre pattern. The super-resolution 
sub-package of PyFibreBundle provides the ability to combine multiple images and generate an enhanced resolution 
using triangular linear interpolation. As with single image triangular linear interpolation, calibation takes
several seconds, but reconstruction is fast. For most applications this functionality is best accessed via the 
:doc:`PyBundle<pybundle_class>` class as shown below.

^^^^^^^^^^^^^^^^
Getting Started 
^^^^^^^^^^^^^^^^

Import pybundle and instantiate a PyBundle object::

    from pybundle import PyBundle
    pyb = PyBundle()
    
To use super-resolution we must use the ``TRILIN`` processing method and choose the output image size::

    pyb.set_core_method(pyb.TRILIN)
    pyb.set_grid_size(512)

Then enable super-resolution::

    pyb.set_super_res(True)
    
As for non-super-resolution ``TRILIN``, we provide a calibration image, a uniformly illuminated image (2D numpy array) used to identify core locations::

    pyb.set_calib_image(calibImage)
    
We can also set a background image (2D numpy array) which will be used for core-by-core background subtraction, this may be the same as ``calibImage``::

    pyb.set_background(backgroundImg)
    
A normalisation image (2D numpy array) for core-by-core normalisation is set using::

    pyb.set_normalise_image(calibImg)   
        
We also need to provide a set of calibration images from which the shifts can be determined. These can be the same
as the images we wish to use to create a super-resolution image from, or a set of image that we know have the same shifts. The calibration
images are provided as a 3D numpy array, with image number along the third axis::

    pyb.set_sr_calib_images(srCalibImages)
           
We then perform the calibration, this may take several seconds::
    
    pyb.calibrate_sr()
        
We can then generate a super-resolution image from a stack of shifted input images, again a 3D numpy array. This may be the same as ``srCalibImages`` or, if the shifts are reproducible, we can re-use the calibration for different stacks of images.::

    srImage = pyb.process(inputImages)  
    
^^^^^^^^^^^^^^^^^^
Using Known Shifts
^^^^^^^^^^^^^^^^^^

If the x and y shifts between images are known in advance, they can be specified using::

    pyb.set_sr_shifits(shifts)

where ``shifts`` is a 2D numpy array of size (nImages, 2). When ``pyb.calibrate_sr()`` is called, these shifts will be used instead of calculating shifts, i.e. ``pyb.set_sr_calib_images()`` does not need to be called.

    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Correcting for Intensity Difference Between Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 
If the shifted images are created simply by moving the bundle or the object, then the above method using only a 
single background/normalisation image is all that is requried. If the shifted images have different intensities 
(e.g. they are created from different light sources) then any global intensity differences must be
corrected to avoid image artefacts. A simple way to correct this is to adjust each shifted image so that the corresponding
calibration image has the same mean intensity. This happens by default cand can be explicitly turned on or off using::

    pyb.set_sr_norm_to_images(True)   
    
or::

    pyb.set_sr_norm_to_images(False)   

Alternatively, if a set of background images with the same relative mean intensities is available, these can be used
to determine the required correction by setting::

    pyb.set_sr_norm_to_backgrounds(True)   
    pyb.set_sr_backgrounds(backgroundImgs)
           
Note that this 'normalisation' is correcting for global differences in the intensity of each of the shfited images and 
is distinct from the core-by-core normaliation of the TRLIN method which corrects for core-to-core variations.
    
These options must be set prior to calibration.    
    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using Background/Normalisation Stack 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If each shifted image requires a different core-by-core background subtraction and/or normalisation, this can be specified::

    pyb.set_sr_multi_backgrounds(True)
    pyb.set_sr_backgrounds(backgroundImgs)
    
    pyb.set_sr_multi_normalisation(True)
    pyb.set_sr_normalisation_images(normalisationImgs)
    
In this case, a different TRILIN normalisation/background correction is applied to each shifted image independently. If multi-normalisation is used, this overrides ``set_sr_norm_to_backgrounds`` or ``set_sr_norm_to_images`` since it will inherently ensure that
each image is corrected to be of the same mean intensity.

These options must be set prior to calibration.    

    
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using Lower Level Functions 
^^^^^^^^^^^^^^^^^^^^^^^^^^^        
   
First, perform the calibration. This requires a flat-field/background image ``calibImg`` (a 2D numpy array), a stack of shifted images ``imgs`` (a 3D numpy array - [x,y,n]), an estimate of the core spacing ``core size``, and the output image size ``gridSize`` ::

    calib = SuperRes.calib_multi_tri_interp(calibImg, imgs, coreSize, gridSize, normalise = calibImg)

We have also specified an optional parameter, a normalisation image ``calibImg``, which prevents the images becoming grainy due to core-core variations. Note that ``imgs`` does not need to be the actual images to be used for reconstruction, but they must have the same relative shift as the the images. Alternatively, if the shifts are known, these can be specified using the optional parameter ``shifts`` which should be a 2D numpy array of the form (x_shift, y_shift, image_number). If ``shifts`` is specified then ``imgs`` can be ``None``.

We then perform the super-resolution reconstruction using::

    reconImg = SuperRes.recon_multi_tri_interp(imgs, calib)

which returns ``reconImg`` a 2D numpy array representing the output image.

Additional options are described below.


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
* ``multiBackgrounds`` If ``True`` each shifted image will have an independent core-background subtraction based on the background images provided in ``backgroundImgs``.
* ``multiNormalisation`` If ``True`` each shifted image will have an independent core-normalisation based on the normalisation images provided in ``normalisationImgs``.
* ``backgroundImgs`` Stack of images, same size as ``imgs`` which are used either for correcting mean image intensity (if ``normToBackground`` used) or image-by-image core background correction (if ``multiBackgrounds`` used).
* ``normalisationImgs`` Stack of images, same size as ``imgs``, which is used for image-by-image core normalisation.
* ``imageScaleFactor`` If normToBackground and normToImage are False (default), use this to specify the normalisation factors for each image. Provide a 1D array the same size as the number of shifted images. Each image will be multiplied by the corresponding factor prior to reconstruction. Default is None (i.e. no scaling).
* ``postFilterSize`` Sigma of Gaussian filter applied to image after reconstruction, defaults to no filter.
* ``autoMask`` Whether to mask pixels outside bundle when searching for cores. Defualts to ``True``.
* ``mask`` Whether to mask pixels outside of bundle in reconstructed image. Defaults to ``True``.

*Returns:*

* Instance of ``BundleCalibration``

.. py:function:: recon_multi_tri_interp(imgs, calib, [useNumba])

*Required arguments:*

* ``imgs`` Stack of shifted images as 3D numpy array. The third axis is image number.
* ``calib`` Instance of ``bundleCalibration`` returned by ``calib_multi_tri_interp``.

*Optional arguments:*

* ``useNumba`` Boolean, whether to use Numba package to speed up reconstruction if available. Defaults to ``true``.

*Returns:*

* Reconstructed image as 2D numpy array.

.. py:function:: sort_sr_stack(stack, stackLength)

A helper function that takes a stack of images and extracts an ordered set of images relative to a reference 
'blank' frame which is much lower intensity than the other frames.  For use with super-resolution systems
which use a blank frame as a reference point.

The blank frame can be anywhere in the stack, and the output stack will be formed cyclically
from frames before and after the blank frame. For example, if we have frames

       1  2  3  X  4  5

where X is the blank frame, the function will return a stack in the following order

       4  5  1  2  3
               
The input stack, ``stack`` should should have ``stackLength + 1``  frames and 
there must be ``stackLength + 1`` images in each cycle  (i.e. ``stackLength`` useful 
images plus one blank reference image).

The blank reference image is not returned, i.e the returned stack has ``stackLength`` frames.
    
Input stack should have frame number in third dimension.

*Required arguments:*

* ``stack`` Input images (x,y,frame_num), a stack containing (stackLength + 1) frames, one of which is blank.
* ``stackLength`` Desired number of images in output stack.

*Returns:*

* Re-arranged stack.

^^^^^^^^^^^^^^^^^^^^^^
Implementation Details 
^^^^^^^^^^^^^^^^^^^^^^

``calib_multi_tri_interp`` first calls ``calib_tri_interp`` to perform the standard calibration for triangular linear interpolation. This obtains the core locations, using ``find_cores``. If the optional parameters ``normToImage`` or ``normToBackground`` are set to ``True``, then the mean image intensity for ether the images stack or the background stack (supplied as a further optional parameter ``backgroundImgs``) are calculated and stored. These are then later used to normalise each of the input images to a constant mean intensity. This is important for applications where the illumination intensity will be different for each image, but in most applications would not be needed. It is also possible to provide a 1D array of normalisation factors directly as the ``imageScaleFactor`` parameter.

``calib_multi_tri_interp`` then calculates the relative shifts between the supplied images in ``imgs`` using ``get_shifts`` via normalised cross correlation. Alternatively, shifts can be provided via the optional parameter ``shifts``. For each image, the recorded core positions are then translated by the measured shifts, and a single list of shifted core positions is assembled, containing the shifted core positions from all the images. ``init_tri_interp`` is then called, which forms a Delaunay triangulation over this set of core positions. For each pixel in the reconstruction grid the enclosing triangle is identified and the pixel location in barycentric co-ordinates is recorded.

Reconstruction is performed using ``recon_multi_tri_interp``.  The intensity value from each core in each of the images are extracted, and then pixel values in the final image are interpolated linearly from the three surrounding (shifted) cores, using the pre-calculated barycentric distance weights.

The SR calibration is stored in an instane of ``BundleCalibration``. This is an extension of the regular TRILIN calibration, and so this super-resolution calibration can be used for non-super-resolution reconstructions (but not vice-versa).

^^^^^^^^
Examples
^^^^^^^^

Examples are provided in "examples\super_res_oop_example" for use via PyBundle class, and "examples\\super_res_example.py" for calling lower-level functions directly.
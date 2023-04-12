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
corrected to avoid image artefacts. A simple way to correct this is by multiplying each shifts images by an intensity
correction factor such that all the shifted images in such a way that multipying all the corresponding
calibration images by the same set of factors would results in them all having the same mean intensity. 
This happens by default and can be explicitly turned on or off using::

    pyb.set_sr_norm_to_images(True)   
    
or::

    pyb.set_sr_norm_to_images(False)   

Alternatively, if a set of background images with the same relative mean intensities is available, these can be used
to determine the required correction by setting::

    pyb.set_sr_backgrounds(backgroundImgs)
    pyb.set_sr_norm_to_backgrounds(True)   
    
where ``backgroundImgs`` is a 3D numpy array of (height, width, num images) containing the set of background images.
           
Note that this 'normalisation' is correcting for global differences in the intensity of each of the shfited images and 
is distinct from the core-by-core normaliation of the TRILIN method which corrects for core-to-core variations.
    
These options must be set prior to calibration.    
    
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using Background/Normalisation Stack 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If each shifted image requires a different core-by-core background subtraction and/or normalisation, this can be specified::

    pyb.set_sr_multi_backgrounds(True)
    pyb.set_sr_backgrounds(backgroundImgs)
    
    pyb.set_sr_multi_normalisation(True)
    pyb.set_sr_normalisation_images(normalisationImgs)

In this case, a different TRILIN normalisation/background correction is applied to each shifted image independently. 
If multi-normalisation is used, this overrides ``set_sr_norm_to_backgrounds`` or ``set_sr_norm_to_images`` since it will inherently ensure that
each image is corrected to be of the same mean intensity. This may offer an improvement over correcting on the basis of the mean background image intensities (i.e. using ``pyb.set_norm_to_backgrounds``)   
in cases where the coupling efficiency of individual cores varies across the set of shifted images.     

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


^^^^^^^^^^^^^^^^^^^^^^^^^^^
Parameterised Shifts
^^^^^^^^^^^^^^^^^^^^^^^^^^^  

In some circumstances, the shifts between the images in the stack are fixed in time but are linearly dependent on some other parameter. 
For example, in fibre bundle inline holographic microscopy, which uses multiple light sources in a transmission geometry, 
the shifts of the hologram (image) position on the bundle depend on the distance between the object and the 
bundle. In these cases it can be convenient to determine the dependence of the shifts on this parameter in a calibrations stage, and then to 
subsequently infer the shifts for all further sets of images based on the current value of the parameter rather than measuring them directl from the images.

Assuming we have acquired several stacks of shifted images for different values of the parameter, 
we assemble a 4D numpy array of images with dimensions (image height, image width, number of shifts, number of example param values).

We then call::
    
    paramCalib = calib_param_shift(param, images, calibration)
    
where ``calibration`` is a single image linear interpolation calibration such as returned from ``calib_tri_interp`` or the one stored
in ``PyBundle.calibration`` after calling ``PyBundle.calibrate``. ``param`` is a 1D numpy array specifying the values of the parameter
for each stack of shifted images. 

The calibration is used to reconstruct each image in the stack. The x and y-components of the shifts of the nth shifted image 
from each stack of shifted images (i.e. across all example values of the parameter) are then fitted to the values of the parameter with 
a 1st order polynomial. The function returns ``paramCalib`` which provides the gradient and offset of the shift for each image 
in the stack of shifted images.

To calculate the expected shifts for a stack of shifted images for a specific value of the parameter, we then call::

    shifts = get_param_shift(param, paramCalib)
    
Where ``param`` is the value of the parameter we wish to know the shifts for, and ``paramCalib`` is the calibration returned by ``calib_param_shift``.


^^^^^^^^^^^^^^^^^^^^^^^^^
Calibration Look-up-table
^^^^^^^^^^^^^^^^^^^^^^^^^

In cases where the shifts between the images change in some linear way with some parameter, as discussed in detail above, it may be desirable
to reconstruct resolution-enhanced images for multiple values of the parameter. For example, in inline bundle holographic microscopy, the
shifts depend on the distance to the object which may change in time. A different value for the shifts requires a new SR calibration,
since the calibration requires knowledge of the shifts. However, this calibration is too slow to be performed in real-time. It is therefore
advantageous to compute different calibrations for different values of the parameter, store these in a look-up table (LUT), and then at run-time 
to use the calibration stored for the nearest value of the parameter.

To generate a LUT when using the PyBundle class, call::

    PyBundle.calibrate_sr_LUT(self, paramCalib, paramRange, nCalibrations) 
    
Here, ``paramRange`` is a tuple of (min, max) values of the parameter to generate the LUT for, and ``nCalibrations`` is the number of 
values of the parameter within this range to create calibrations for. 
Since each calibration takes typically several seconds to perform, large values of ``nCalibrations`` will take a long time to compute.
    
Prior to calling ``calibrate_sr_LUT``, a single image calibration must already have been created using ``pyb.calibrate``. 
We must also provide a parameter calibration ``paramCalib``, which tells us how the image shifts are related to the value of the parameter, 
either created manually or using the output of ``calib_param_shift``. 

We then tell PyBundle to use the LUT::

    PyBundle.set_use_sr_lut(True)
    
We must also tell PyBundle the current value of the parameter::

    PyBundle.set_sr_param(paramValue)
      
Now, when we call ``PyBundle.process``, assuming we have enabled super-resolution and provided a set of shifted images, as described above,
the calibration LUT will be accessed and the calibration previously created for a value of the parameter closest to the current value will be used. 
This look up is much faster (by several orders of magnitude) then performing a new calibration.

Alternatively, if lower-level control is needed, an instance of the CalibrationLUT can be created directly::

    lut = CalibrationLUT(calibImg, imgs, coreSize, gridSize, paramCalib, paramRange, nCalibrations, [optional arguments])
    
Parameters (including optional parameters) are as for ``calib_multi_tri_interp``, with the addition of ``paramCalib``, which is the return 
from calling  ``calib_param_shift`` and stores the mapping between the parameter and the shifts, ``paramRange`` which is a tuple of (min, max) 
values of the parameter to generate the LUT for, and ``nCalibrations`` which is the number of values of the parameter within this range to 
create calibrations for. 

Once the LUT is generated, we can extract the best calibration for the current value of the parameter using::
  
    calibrationSR = lut.calibrationSR(paramValue)

SR reconstructions can then be performed using::

    reconImage = recon.multi_tri_interp(imageStack, calibrationSR)




^^^^^^^^^^^^^^^^^^
Function Reference
^^^^^^^^^^^^^^^^^^

These are the low level functions, for most purposes it is better to use an instance of the ``PyBundle`` class.

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



.. py:function:: multi_tri_backgrounds(calibIn, backgrounds) 

Updates a multi_tri calibration with a new set of backgrounds without requiring full recalibration

*Required arguments:*

* ``calibIn`` super-resolution bundle calibration, instance of BundleCalibration
* ``backgrounds``: stack of background images, 3D numpy array with image number on 3rd axis

*Returns:*

* An instance of BundleCalibration which contatins the updated background values


    
.. py:function:: calib_param_shift(param, images, calibration)

For use when the shifts between the images are linearly dependent on some other parameter. 

* ``param`` a 1D numpy array containing example values of the parameter for calibration
* ``images`` a set of shifted image stacks, one for each example value of the parameter. Provide the images as a 4D numpy array of (y, x, shift, parameter).
* ``calibation`` a single image bundle calibration as an instance of BundleCalibration

*Returns:*

* A 3D array of calibration factors, giving the gradient and offset of x and y shifts of each image with respect to the parameter.
           



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

Examples are provided in "examples\\super_res_example" for use via PyBundle class, and "examples\\super_res_example_low_level.py" for calling lower-level functions directly.
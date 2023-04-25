Mosaicing
====================================
The Mosaic class allows high speed mosaicing using normalised cross correlation to detect shifts between image frames, 
and either dead-leaf or alpha-blended insertion of images into a mosaic. 
The easiest way to use this functionality is to create an instance of ``Mosaic`` class and then use ``Mosaic.add(img)`` to 
sequentially register and add image ``img`` to the mosaic,	 and ``Mosaic.getMosaic()`` to get the latest mosaic image. 
Both ``img`` and the ``mosaic`` are 2D (monochrome) or 3D (colour) numpy arrays.

^^^^^^^^^^^^^^^
Getting Started
^^^^^^^^^^^^^^^

Instantiate an object of the ``Mosaic`` class using default options and with a mosaic image size of 1000x1000::

    mMosaic = Mosaic(1000)

Add an image ``img`` to the mosaic::

    mMosaic.add(img) 

Request the latest mosaic image::

    mosaicImage = mMosaic.getMosaic()

The ``mosaicImage`` will be a 2D numpy array if ``img`` is 2D and a 3D numpy array if ``img`` is 3D, in which case the third channel represents the colour channels.	


^^^^^^^^^^^^^
Instantiation
^^^^^^^^^^^^^

.. py:function:: Mosaic(mosaicSize [, resize=None, imageType=None, templateSize=None, refSize = None, cropSize = None, blend = True, minDistForAdd = 5, currentX = None, currentY = None, boundaryMethod = CROP, expandStep = 50, resetThresh = None, resetIntensity = None, resetSharpness = None])


*Required arguments:*

* ``mosaicSize`` Size of mosaic image in pixels. This may later change depending on which ``boundaryMethod`` is set.

*Optional arguments:*

* ``resize`` Images are resized to a square of this side-length before insertion (default is same as size of first image added to mosaic, i.e. no resizing).
* ``imageType`` Image type for mosaic image (default is same as first image added to mosaic).
* ``templateSize`` Size of ROI taken from second image when calculating shift between two images. (default is 1/4 of size of first image added).
* ``refSize`` Size of ROI taken from first image when calculating shift between two images. (default is 1/2 of size of first image added).
* ``cropSize`` Input images are cropped to a circle of this diameter before insertion. (default is 0.9 x size of first image added).
* ``blend`` If ``True``, uses distance-weighted alpha blending (default), if ``False`` uses dead-leaf.
* ``blendDist`` If using alpha blending, determines how strong the distance weighting is (default = 40 pixels).
* ``minDistForAdd`` Only add an image to the mosaic if it has moved this far in pixels since last image was added (default = 5 pixels).
* ``currentX`` Initial x position of first image in mosaic (default is centre of image).
* ``currentY`` Initial y position of first image in mosaic (default is centre of image).
* ``boundaryMethod`` Determines what happens when edge of mosaic image is reached. ``Mosaic.CROP`` [Default]: new images go out of mosaic image, ``Mosaic.EXPAND``: mosaic image is made larger, ``Mosaic.SCROLL``: mosaic image is scrolled, with loss of data on the opposite edge.
* ``expandStep`` If boundaryMethod is ``Mosaic.EXPAND``, mosaic will be expanded by this amount when the edge is reached (default is 50).
* ``resetThresh`` If set to value other than None (default), mosaic will reset when correlation between two frames is below this value.
* ``resetIntensity`` If set to value other than None (default), mosaic will reset when mean intensity of a supplied frame is less than this value.
* ``resetSharpness`` If set to value other than None (default), mosaic will reset when sharpness (image gradient) of a supplied frame is less than this value.

^^^^^^^^^^^^^^^^^^^^
Function Reference
^^^^^^^^^^^^^^^^^^^^

.. py:function:: add(img) 
Adds an image ``img`` to the current mosaic.

.. py:function:: get_mosaic() 
Returns a copy of the current mosaic as a 2D/3D numpy array.

^^^^^^^^^^^^^^^^^^^^
Usage Notes
^^^^^^^^^^^^^^^^^^^^
The only required argument is the size of the mosaic image. By default images will be added blended, there will be no resize of the input image, no checking of input image quality and if the mosaic reaches the edge of the image it will simple run off the the edge.

Usually it is beneficial to resize the input images to prevent the need for a very large mosaic image, e.g.::

    mMosaic = Mosaic(1000, resize = 250)

The reset methods (``resetThresh``, ``resetIntensity`` and ``resetSharpness``) are normally required when used with a handheld probe to handle instances where tissue contact is lost or the probe is moved too quickly. For optical sectioning endomicroscope, a combination of correlation based thresholding (``resetThresh``) and intensity based thresholding (``resetIntensity``) works well. For non-sectioning endomicroscopes, moving out of focus does not sufficiently reduce either, and so it may be necessary to use sharpness thresholding (``resetSharpness``) as well. The best values to use must be determined empirically and will depend on pre-processing steps.

For slow moving probes, ``minDistForAdd`` may need to be adjusted particularly when using blending to prevent undesirable effects of the same image being blended with itself.



^^^^^^^^^^^^^^^^^^^
Low Level Functions
^^^^^^^^^^^^^^^^^^^
These functions are used internally and would normally not need to be called directly. Check the source for arguments.

* ``initialise`` This is called the first time an image is added using ``add``. It cannot be called beforehand since some details of the images, such as the size, are required.
* ``find_shift`` Computes shift between two images using normalised cross correlation.
* ``insert_into_mosaic`` Adds an image to the mosaic dead leaf.
* ``insert_into_mosaic_blended`` Adds an image to the mosaic with distance-weighted alpha-blending.
* ``cosine_window`` Generates circular cosine window, used in blending.
* ``is_outside_mosaic`` Returns true if intended image insert will go outside of mosaic image.
* ``expand_mosaic`` Expands mosaic image.
* ``scroll_mosaic`` Scrolls the mosaic image.

^^^^^^^
Example
^^^^^^^

An example is provided in "examples\\mosaicing_example.py"
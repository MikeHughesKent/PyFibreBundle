Mosaicing
====================================
The Mosaic class allows high speed mosaicing using normalised cross correlation to detect shifts between image frames, and either dead-leaf or alpha-blended insertion of images into a mosaic. The easiest way to use this functionality is to create an instance of ``Mosaic`` class and then use ``Mosaic.add(img)`` to sequentially register and add image ``img`` to the mosaic,	 and ``Mosaic.getMosaic()`` to get the latest mosaic image. Both ``img`` and the ``mosaic`` are 2D numpy arrays.

^^^^^^^^^^^
Quickstart
^^^^^^^^^^^

Instantiate an object of the ``Mosaic`` class using default options and with a mosaic image size of 1000x1000::

    mMosaic = Mosaic(1000).

Add an image ``img`` to the mosaic::

    mMosaic.add(img) 

Request the latest mosaic image::

    mosaicImage = mMosaic.getMosaic()
	


^^^^^^^^^^^^^
Instantiation
^^^^^^^^^^^^^

.. py:function:: Mosaic(mosaicSize [, resize=None, imageType=None, templateSize=None, refSize = None, cropSize = None, blend = False, minDistForAdd = 5, currentX = None, currentY = None, boundaryMethod = CROP, expandStep = 50])


*Required arguments:*

* ``mosaicSize`` Size of mosaic image in pixels. This may later change depending on which ``boundaryMethod`` is set.

*Optional arguments:*

* ``resize`` Images are resized to a square of this side-length before insertion (default is same as size of first image added to mosaic, i.e. no resizing).
* ``imageType`` Image type for mosaic image (default is same as first image added to mosaic).
* ``templateSize`` Size of ROI taken from second image when calculating shift between two images. (default is 1/4 of size of first image added).
* ``refSize`` Size of ROI taken from first image when calculating shift between two images. (default is 1/2 of size of first image added).
* ``cropSize`` Input images are cropped to a circle of this diameter before insertion. (default is 0.9 x size of first image added).
* ``blend`` If ``True``, uses distance-weighted alpha blending, if ``False`` uses dead-leaf (default).
* ``blendDist`` If using alpha blending, determines how strong the distance weighting is (default = 40 pixels).
* ``minDistForAdd`` Only add an image to the mosaic if it has moved this far in pixels since last image was added (default = 5 pixels).
* ``currentX`` Initial x position of first image in mosaic (default is centre of image).
* ``currentY`` Initial y position of first image in mosaic (default is centre of image).
* ``boundaryMethod`` Determines what happenswhen edge of mosaic image is reached. ``Mosaic.CROP`` [Default]: images go out of image, ``Mosaic.EXPAND``: mosaic image is made larger, ``Mosaic.SCROLL``: mosaic image is scrolled, with loss of data on the opposite edge.
* ``expandStep`` If boundaryMethod is ``Mosaic.EXPAND``, mosaic will be expanded by this amount when the edge is reached (default is 50).

^^^^^^^^^^^^^^^^^^^^
Methods
^^^^^^^^^^^^^^^^^^^^

.. py:function:: add(img)

Adds an image ``img`` to the current mosaic. ``img`` should be a 2D numpy array.


.. py:function:: get_mosaic()

Returns a copy of the current mosaic as a 2D numpy array.



^^^^^^^^^^^^^^^^^^^
Methods - Low Level
^^^^^^^^^^^^^^^^^^^
These static methods are used internally and would normally not need to be called. Check the source for arguments.

* ``initialise`` This is called the first time an image is added using ``add``. It cannot be called beforehand since some details of the images, such as the size, are required.
* ``find_shift`` Computes shift between two images using normalised cross correlation.
* ``insert_into_mosaic`` Adds an image to the mosaic dead leaf.
* ``insert_into_mosaic_blended`` Adds an image to the mosaic with distance-weighted alpha-blending.
* ``cosin_window`` Generates circular cosine window, used in blending.
* ``is_outside_mosaic`` Returns true if intended image insert will go outside of mosaic image.
* ``expand_mosaic`` Expands mosaic image.
* ``scroll_mosaic`` Scrolls the mosaic image.
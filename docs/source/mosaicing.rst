Mosaicing
====================================
The Mosaic class allows high speed mosaicing using normalised cross correlation to detect shifts between image frames, 
and either dead-leaf or alpha-blended insertion of images into a mosaic. 
The easiest way to use this functionality is to create an instance of ``Mosaic`` class and then use ``Mosaic.add(img)`` to 
sequentially register and add image ``img`` to the mosaic,	 and ``Mosaic.getMosaic()`` to get the latest mosaic image. 
Both ``img`` and the ``mosaic`` are 2D (monochrome) or 3D (colour) numpy arrays.

An example is provided on `Github <examples\\mosaicing_example.py>`_.

^^^^^^^^^^^^^^^
Getting Started
^^^^^^^^^^^^^^^

Instantiate an object of the ``Mosaic`` class using default options and with a mosaic image size of 1000x1000::

    from pybundle import Mosaic
    mMosaic = Mosaic(1000)

Add an image ``img`` to the mosaic::

    mMosaic.add(img) 

Request the latest mosaic image::

    mosaicImage = mMosaic.getMosaic()

The ``mosaicImage`` will be a 2D numpy array if ``img`` is 2D and a 3D numpy array if ``img`` is 3D, in which case the third axis represents the colour channels.	


^^^^^^^^^^^^^^^^^^^^
Methods
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pybundle.Mosaic
   :members:
   
   
   
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
The private member functions of the Mosaic class are listed below for custom use:

.. autoclass:: pybundle.Mosaic
   :noindex:
   :private-members: __initialise_mosaic, __insert_into_mosaic, __insert_into_mosaic_blended, __find_shift, __cosine_window, __is_outside_mosaic, __expand_mosaic, __scroll_mosaic
   

^^^^^^^
Example
^^^^^^^

An example is provided in "examples\\mosaicing_example.py"
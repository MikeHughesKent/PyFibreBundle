------------------
Function Reference
------------------
A list of lower-level functions is available below, see the documentation for individual
classes for class methods.

The `PyBundle class <pybundle_class.html>`_ implements most of the functionality of the package and is the preferred approach for most applications except for Mosaicing, which is handled
by the `Mosaic class <mosaicing.html>`_. 

PyFibreBundle uses numpy arrays as images throughout, wherever 'image' is specified this refers to a 2D (monochrome) or 3D (colour) numpy array. For colour images, the colour channels are along the third axis. There can be as many colour channels as needed.


^^^^^^^^^^^^^^
Classes
^^^^^^^^^^^^^^

.. py:function:: PyBundle()
   :noindex:


Provides object-oriented access to core functionality of PyFibreBundle. 
See `PyBundle class <pybundle_class.html>`_ for details.

.. py:function:: Mosaic()
   :noindex:

Provides object-oriented access to mosaicing functionality of PyFibreBundle. 
See `Mosaic class <mosaicing.html>`_ for details.




^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Low-Level Functions for Bundle finding, cropping, masking
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: pybundle.auto_mask


.. autofunction:: pybundle.auto_mask_crop


.. autofunction:: pybundle.apply_mask


.. autofunction:: pybundle.crop_rect


.. autofunction:: pybundle.find_bundle


.. autofunction:: pybundle.find_core_spacing


.. autofunction:: pybundle.get_mask





^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Low Level Functions for Spatial Filtering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pybundle.g_filter


.. autofunction:: pybundle.crop_filter_mask


.. autofunction:: pybundle.edge_filter


.. autofunction:: pybundle.filter_image


.. autofunction:: pybundle.median_filter


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Functions for Triangular Linear Interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
""""""""""""""""""""
High-level functions
""""""""""""""""""""

.. autofunction:: pybundle.calib_tri_interp

.. autofunction:: pybundle.recon_tri_interp


"""""""""""""""""""
Low-level functions
"""""""""""""""""""

.. autofunction:: pybundle.find_cores

.. autofunction:: pybundle.core_values

.. autofunction:: pybundle.init_tri_interp



^^^^^^^^^^^^^^^^^
Utility Functions
^^^^^^^^^^^^^^^^^

.. autofunction:: pybundle.average_channels

.. autofunction:: pybundle.extract_central

.. autofunction:: pybundle.max_channels

.. autofunction:: pybundle.radial_profile

.. autofunction:: pybundle.save_image8

.. autofunction:: pybundle.save_image8_scaled

.. autofunction:: pybundle.save_image16

.. autofunction:: pybundle.save_image16_scaled

.. autofunction:: pybundle.to8bit

.. autofunction:: pybundle.to16bit




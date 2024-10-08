PyFibreBundle
====================================
PyFibreBundle is a Python package for processing of images captured through optical fibre bundles. 

The project is hosted on `github <https://github.com/MikeHughesKent/PyFibreBundle/>`_. The latest stable release can be installed via pip::

    pip install PyFibreBundle

The package supports fibre core pattern removal by filtering and triangular linear interpolation, background correction and 
flat fielding, as well as automatic bundle location, cropping and masking. Both monochrome and colour images can be processed.
The :doc:`PyBundle<pybundle_class>` class is the preferred way to access this functionality, 
but the lower level functions can also be used directly for greater customisation. 
The :doc:`Mosaic<mosaicing>` class provides mosaicing via normalised cross correlation, 
and the :doc:`SuperRes<super_res>` class allows multiple shifted images to be combined to improve resolution.

The package is designed to be fast enough for use in imaging GUIs as well as for offline research - 
frame rates of over 100 fps can be achieved on mid-level hardware, including core removal and mosaicing. 
The Numba just-in-time compiler is used to accelerate key portions of code (particularly triangular linear interpolation) and 
OpenCV is used for fast mosaicing. If the Numba package is not installed then PyFibreBundle falls back on Python interpreted code.

PyFibreBundle is developed mainly by `Mike Hughes <https://research.kent.ac.uk/applied-optics/hughes/>`_'s lab in the `Applied Optics Group <https://research.kent.ac.uk/applied-optics>`_, School of Physics and Astronomy, University of Kent. Bug reports, contributions and pull requests are welcome. 

^^^^^^^^
Contents
^^^^^^^^

.. toctree::
   :maxdepth: 2
   
   install
   core
   linear_interp
   pybundle_class
   mosaicing
   super_res
   low_level
   functions
   
* :ref:`genindex`


*Acknowledgements: Cheng Yong Xin, Joseph, contributed to triangular linear interpolation; Petros Giataganas who developed some of the Matlab code that parts of this library were ported from. Funding to Mike Hughes's lab from EPSRC (Ultrathin fluorescence microscope in a needle, EP/R019274/1), Royal Society (Ultrathin Inline Holographic Microscopy).*   






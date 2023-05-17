----------------------
Installation
----------------------

There are several ways to get PyFibreBundle:

* Download the `latest stable release <https://github.com/MikeHughesKent/PyFibreBundle/releases/latest>`_ from github and unzip.  (The download link is at the bottom of the linked page.)
This will give you all the examples, tests and test data.

* Download the latest files from the `Github repository <https://github.com/MikeHughesKent/PyFibreBundle>`_ by clicking 'Code' and 'Download ZIP'

* Clone the `Github repository <https://github.com/MikeHughesKent/PyFibreBundle>`_ using::

    git clone https://github.com/MikeHughesKent/PyFibreBundle 
    
* Install the latest stable release using::

    pip install PyFibreBundle 


Using pip install should find and install all the dependencies. For the other 
options you will need to either manually check you have the requirements 
installed, or navigate to the PyFibreBundle folder on your machine and run::

    pip install -r requirements.txt
    
to install the dependencies. You may wish to create a virtual environment 
using Conda/venv first to avoid conflicts with your existing python setup.

Note that the pip install doesn't include the examples and tests which still 
need to be downloaded from Github.

Once installed, you can try running the examples in the examples folder. The 
examples assume they are being run from the working directory.

^^^^^^^^^^^^
Dependencies
^^^^^^^^^^^^

* matplotlib>=3.3.4
* numba>=0.55.1
* numpy>=1.18
* opencv_python>=4.5.2.54
* Pillow>=9.3.0
* scipy>=1.7.3


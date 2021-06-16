# PyBundle
Python library for image processing fibre bundle images

Currently under development. Mostly uses OpenCV library. The aim is to keep this fast enough for use in real-time image acquisition and display system.

## Features currently implemented
* Locate bundle in image
* Crop image to only show bundle
* Mask areas outside of bundle
* Gaussian spatial filtering to remove core pattern
* Find centers of all cores in bundle (two implementations: regional maxima and Hough transform)

## Currently being worked on
* Triangular linear interpolation to remove core pattern.



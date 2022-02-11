# PyBundle
PyBundle is a Python library for basic image processing of fibre bundle images. It is being developed mainly by Mike Hughes at the Applied Optics Group, University of Kent, but pull requests are welcome.

This library is currently under development and there is no stable release. Some functions require the OpenCV library. The aim is to keep this fast enough for use in real-time image acquisition and display systems.

__Acknowledgements__: Cheng Yong Xin, Joseph, contributed to triangular linear interpolation; Petros Giataganas who developed some of the Matlab code that parts of this library were ported from. Funding from EPSRC (Ultrathin fluorescence microscope in a needle, EP/R019274/1), Royal Society (Ultrathin Inline Holographic Microscopy).

## Bundle Processing  
* Locate bundle in image
* Crop image to only show bundle
* Mask areas outside of bundle
* Gaussian spatial filtering to remove core pattern
* Find centers of all cores in bundle (two implementations: regional maxima and Hough transform)
* Core removal using trinagular linear interpolation following Delaunay triangulation. Currently quite slow.

## Mosaicing
* Detect image to image shift using normalised cross correlation.
* Insert image into mosaic either using dead-leaf or alpha blending.
* Expand or scroll mosaic when the edge of the mosaic image is reached.

## Currently being worked on
* Speed improvement to triangular linear interpolation to remove core pattern. 

## PyBundle Class
PyBundle functions are static.

### Bundle locating, masking and filtering
* __gFilter__(img, filterSize) : Applies 2D Gaussian filter of sigma *filterSize* to *img*.
* __findBundle__(img, [filterSize = 4]) : Finds bundle in image *img* using preprocessing Gaussian filter of sigma *filterSize*. Returns tuple of *loc = (centreX, centreY, radius)*.
* __getMask__(img, loc) : Returns a mask as a 2D numpy array which is 1 inside bundle and 0 outside, using bundle location tuple *loc = (centreX, centreY, radius)*. 
* __applyMask__(img, mask) : Mutiplies image *img* with mask *mask*. *mask* is a 2D numpy array of the same size as image.
* __mask__(img, loc) : Generates a mask based on tuple *loc = (centreX, centreY, radius)* and then applies it to image *img*. 
* __cropRect__(img, loc) : Extracts square region of interest around bundle in image *img* defined by tuple *loc = (centreX, centreY, radius)*. Returns tuple of *(newImage, newLoc)* where *newLoc* is a tuple of *(centreX, centreY, radius)* adjusted to still be corrected after the cropping.
* __cropFilterMask__(img, loc, mask, filterSize) : Combined processing using previously located bundle at tuple *loc = (centreX, centreY, radius)* and previously calculated *mask*. Crops the *img* to a square around the bunldle, applies Gaussian filter of sigma *filterSize* and sets pixels outside bundle radius to 0.

### Core Finding
* __findCores__(img, coreSpacing) : Finds bundle cores in image *img* where the separation of the cores is *coreSpacing*. Use regional maxima approach and is fast.
* __findCoresHough__(img, kwargs) : Experimental. Finds cores using Hough transform. Currently quite slow and not as accurate as findCores. See source for optional parameters.

### Core Removal
* __calibTriInterp__(img, coreSize, gridSize, [centreX = -1], [centreY = -1], [radius = -1], [filterSize = 0=], [normalise = 0]) : Calibration to allow core removal by triangular linear interpolation. Image *img* should be a uniformly illuminated image of the bundle to allow cores to be located. *coreSize* is the estimated spacing between cores (used by core finding routines). *gridSize* is the number of pixels in the reconstructed images. The reconstructed image will be centred on *(centreX, centreY)* and cover radius *radius*. If any of these parameters are left as the default of -1 then they will be estimated based on the discoverd core positions. Function returns a tuple which is used by *reconTri_Interp*.
* __initTriInterp__ : This is used by calibTriInterp, see source for parameters.
* __reconTriInterp__(img, calib) : Removes core pattern from image *img* using prior calibration *calib* produced by *calibTriInterp*.
* __coreValues__(img, coreX, coreY, filterSize) : Extracts intensity vales from cores in image *img*. *coreX* and *coreY* are vectors specifying locations of cores, *filterSize* is the sigma of a 2D Gaussian filter applied before extracting values (set to 0 for no filter).

### Utility
* __to8bit__(img, [minVal = None], [maxVal = None]) : Returns an image to 8 bit representation. If *minVal* and *maxVa*l are not specified, the minimum and maximum values in the image are scaled to 0 and 255 in the new image. If *minVal* and *maxVal* are specified then these pixel values are mapped to 0 and 255, respectively, and

## Mosaic Class
The Mosaic class allows high speed mosaicing using normalised cross correlation to detect shifts between image frames, and either dead-leaf or alpha blended insertion of images into mosaic. The easiest way to use is to create instance of Mosaic class and then use __Mosaic.add(img)__ to sequentially register and add images to mosaic and __Mosaic.getMosaic()__ to get the latest mosaic image. Both *img* and the mosaic are 2D numpy arrays.

### Initialisation arguments
mMosaic = Mosaic(mosaicSize, optionalArguments).

Required arguments:
* __mosaicSize__ = Size of mosaic image in pixels. This may later change depending on which __boundaryMethod__ is set.

Optional arguments:
* __resize__ = Images are resize to this size before insertion (default = same as size of first image added to mosaic, i.e. no resizing).
* __imageType__ = Image type for mosaic image (default = same as first image added to mosaic).
* __templateSize__ = Size of ROI taken from second image when calculating shift between two images. (default = 1/4 of size of first image added).
* __refSize__ = Size of ROI taken from first image when calculating shift between two images. (default = 1/2 of size of first image added).
* __cropSize__ = Inout images are cropped to a circle of this diameter before insertion. (default = 0.9 x size of first image added).
* __blend__ = If True, uses distance-weighted alpha blending, if False uses dead-leaf.
* __blendDist__ = If using alpha blending, determines how strong the distance weighting us (default = 40 pixels)
* __minDistForAdd__ = Only add an image to the mosaic if it has moved this far in pixels since last image was added (default = 5 pixels).
* __currentX__ = Initial x position of first image in mosaic (default = centre of image).
* __currentY__ = Initial y position of first image in mosaic (default = centre of image).
* __boundaryMethod__ = What to do when edge of mosaic is reached. Mosaic.CROP = images go out of image, Mosaic.EXPAND = mosaic image is made larger, Mosaic.SCROLL = mosaic image is scrolled, with loss of data on the opposite edge. (default - Mosaic.CROP)
* __expandStep__ = If boundaryMethod is Mosaic.EXPAND, mosaic will be expanded by this amount when the edge is reached (default = 50).

### Methods - High Level
* __add(img)__ : Adds 2D numpy array image *img* to mosaic.
* __getMosaic()__ : Returns mosaic image as 2D numpy array.


### Static Methods - Low Level
These static methods are used internally and would normally not need to be called. Check the source for arguments.
* __findShift__ : Computes shift between two images using normalised cross correlation
* __insertIntoMosaic__ : Adds an image to the mosaic dead leaf.
* __insertIntoMosaicBlended__ : Adds an image to the mosaic with distance-weighted alpha-blending.
* __cosinWindow__ : Generates circular cosine window, used in blending.
* __isOutsideMosaic__ : Returns true if intended image insert will go outside of mosaic image.
* __expandMosaic__ : Expands mosaic image.
* __scrollMosaic__ : Scrolls the mosaic image.


# PyBundle
Python library for image processing fibre bundle images. Developed mainly by Mike Hughes at the Applied Optics Group, University of Kent, but pull requests welcome.

Currently under development. Mostly uses OpenCV library. The aim is to keep this fast enough for use in real-time image acquisition and display system.

## Bundle Processing  
* Locate bundle in image
* Crop image to only show bundle
* Mask areas outside of bundle
* Gaussian spatial filtering to remove core pattern
* Find centers of all cores in bundle (two implementations: regional maxima and Hough transform)

## Mosaicing
* Detect image to image shift using normalised cross correlation.
* Insert image into mosaic either using dead-leaf or alpha blending.
* Expand or scroll mosaic when the edge of the mosaic image is reached.

## Currently being worked on
* Triangular linear interpolation to remove core pattern. Delaunay triangulation works, problem is finding which enclosing triangle each pixel is. Matlab has pointLocation for this, but no similar function in OpenCV.

## PyBundle Class
PyBundle functions are static.

### Bundle locating, masking and filtering
* __gFilter__(img, filterSize) : Applies 2D Gaussian filter of sigma *filterSize* to *img*.
* __bundleLocate__(img) : Locates bundle in *img* and returns tuple of (centreX, centreY, radius, mask) where mask is a 2D numpy array of 1 inside bundle and 0 outside.
* __findBundle__(img, filterSize = 4) : Customised location of bundle in *img* using preprocessing Gaussian filter of sigma *filterSize*. Returns tuple of *loc = (centreX, centreY, radius)*.
* __maskAuto__(img, loc) : Locates bundle in *img* and returns an image with pixels outside bundle set to 0.
* __cropRect__(img, loc) : Extracts region of interest around bundle in image *img* defined by tuple *loc = (centreX, centreY, radius)*. 
* __cropFilterMask__(img, loc, mask, filterSize) : Combined processing using previously located bundle at tuple *loc = (centreX, centreY, radius)* and previously calculated *mask*. Crops the *img* to a square around the bunldle, applies Gaussian filter of sigma *filterSize* and sets pixels outside bundle radius to 0.
* __getMask__(img, loc) : Returns a numpy array which is 1 inside bundle and 0 outside, using bundle location tuple *loc = (centreX, centreY, radius)*. 

### Core Finding
* __findCores__(img, coreSpacing) : Finds bundle cores in image *img* where the separation of the cores is *coreSpacing*. Use regional maxima approach and is fast.
* __findCoresHough__(img, kwargs) : Experimental. Finds cores using Hough transform. Currently quite slow and not as accurate as findCores. See source for optional parameters.

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


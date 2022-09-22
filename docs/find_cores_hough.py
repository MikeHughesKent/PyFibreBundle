# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 16:14:04 2022

@author: Applied Optics Group
"""
def find_cores_hough(img, **kwargs):

     scaleFac = kwargs.get('scaleFactor', 2)
     cannyLow = kwargs.get('cannyLow', .05)
     cannyHigh = kwargs.get('cannyHigh', .8)
     estRad = kwargs.get('estRad', 1)
     minRad = kwargs.get('minRad', np.floor(max(1, estRad)).astype('int'))
     maxRad = kwargs.get('maxRad', np.floor(minRad + 2).astype('int'))
     minSep = kwargs.get('minSep', estRad * 2)
     darkRemove = kwargs.get('darkRemove', 2)
     gFilterSize = kwargs.get('filterSize', estRad / 2)

     imgR = cv.resize(img, [scaleFac * np.size(img, 1),
                      scaleFac * np.size(img, 0)]).astype(float)

     # Pre filter with Gaussian and Canny
     imgF = pybundle.g_filter(imgR, gFilterSize*scaleFac)
     imgF = imgF.astype('uint8')
     edges = cv.Canny(imgF, cannyLow, cannyHigh)

     # Using Scikit-Image Hough implementation, trouble getting CV to work
     radii = range(math.floor(minRad * scaleFac),
                   math.ceil(maxRad * scaleFac))
     circs = hough_circle(edges, radii, normalize=True, full_output=False)

     minSepScaled = np.round(minSep * scaleFac).astype('int')

     for i in range(np.size(circs, 0)):
         circs[i, :, :] = np.multiply(circs[i, :, :], imgF)

     accums, cx, cy, radii = hough_circle_peaks(
         circs, radii, min_xdistance=minSepScaled, min_ydistance=minSepScaled)

     # Remove any finds that lie on dark points
     meanVal = np.mean(imgF[cy, cx])
     stdVal = np.std(imgF[cy, cx])
     removeCore = np.zeros_like(cx)
     for i in range(np.size(cx)):
         if imgF[cy[i], cx[i]] < meanVal - darkRemove * stdVal:
             removeCore[i] = 1
     cx = cx[removeCore != 1]
     cy = cy[removeCore != 1]

     cx = cx / scaleFac
     cy = cy / scaleFac

     return cx, cy
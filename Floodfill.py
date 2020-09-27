# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 00:08:07 2020

@author: Harneet singh
"""

import cv2
import numpy as np
import skimage.exposure

# read image
img = cv2.imread('C:/Users/Harneet singh/Desktop/Implementation/Q5/Image for Q5(ii).png')
h, w = img.shape[:2]

# convert to gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold to binary
thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]

# create zeros mask 2 pixels larger in each dimension
mask = np.zeros([h + 2, w + 2], np.uint8)

# floodfill white between two polygons at 240,240
ffimg = thresh.copy()
ffimg = cv2.floodFill(ffimg, mask, (240,240), 255)[1]

# apply distance transform
distimg = ffimg.copy()
distimg = cv2.distanceTransform(distimg, cv2.DIST_L2, 5)

# Maximum spacing between polygons is 2 * largest value in distimg
max = round(2*np.amax(distimg))
print('max spacing:', max)

# Minimum spacing between polygons is 2 * smallest non-zero value in distimg
nonzero_vals = np.nonzero(distimg)
min = round(2*np.amin(nonzero_vals))
print('min spacing:', min)

# scale distance image for viewing
distimg = skimage.exposure.rescale_intensity(distimg, in_range='image', out_range=(0,255))
distimg = distimg.astype(np.uint8)

# save image
cv2.imwrite('polygons_floodfill.png',ffimg)
cv2.imwrite('polygons_distance.png',distimg)

# show the images
cv2.imshow("thresh", thresh)
cv2.imshow("floodfill", ffimg)
cv2.imshow("distance", distimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
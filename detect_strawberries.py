from __future__ import division
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

#### OBJECT DETECTION WITH OPENCV - Siraj Raval
# https://www.youtube.com/watch?v=OnWIYI6-4Ss

### WRITE FIRST THE STEPS AND THEN THE FUNCTIONS

green = (0, 255, 0)

def show(image):
	# figure size in inches
	plt.figure(figsize=(10,10))
	plt.imshow(image, interpolation="nearest")

def overlay_mask(mask, image):
	# actually applying the mask to an image
	rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

	# think of images as arrays
	# adding the weighted sum of the mask and the original image
	# to get the mask overlay
	# converting the MASK from gray to rgb
	img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
	return img

def find_biggest_contour(image):
	# copy the image, so we can modify it
	image = image.copy()

	# this function gives all the
	# we only want the endpoints of the contours, not the whole ones
	# and we'll end up with a list of those contours
	contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	# get all their sizes
	contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
	# isolating the biggest contour
	biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

	# return the biggest contour
	mask = np.zeros(image.shape, np.uint8)
	cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
	return biggest_contour, mask

def circle_contour(image, contour):
	# bounding ellipse
	image_with_ellipse = image.copy()
	ellipse = cv2.fitEllipse(contour)
	# add it
	cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.CV_AA)
	return image_with_ellipse

def find_strawberry(image):
	### STEP 1
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	### STEP 2
	max_dimension = max(image.shape)
	# max dimension we'll use is 700x660 pixels
	# so we scale it to make it smaller than that
	scale = 700/max_dimension
	# scaling the image to be a square
	image = cv2.resize(image, None, fx=scale, fy=scale)

	### STEP 3 - cleaning
	# GaussianBlur always if you want a clean, smooth color
	# kernel size is (7, 7) because the image size is 700x700
	image_blur = cv2.GaussianBlur(image, (7,7), 0)
	# HSV is a color scheme that separates the luma (img intensity)
	# from the chroma (color information). we just want to focus on color
	image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

	### STEP 4 - define our filters
	# filter by color (now separate from the intensity)
	# if more red it's a strawberry, if less red it's not
	# these are hex-sums (what is this?)
	min_red = np.array([0, 100, 80])
	max_red = np.array([10, 256, 256])

	# making a filter
	mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)

	# filter by brightness
	min_red2 = np.array([170, 100, 80])
	max_red2 = np.array([180, 256, 256])

	mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)

	# take these two masks and combine them both

	mask = mask1 + mask2

	### STEP 5 - segmentation
	# separate the strawberry from everything else

	# add elipse around strawberry
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
	# closing operation
	# dilation followed by erosion - helps close small holes in the foreground of small objects
	# make sure it's a smooth red layer
	mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
	# erosion followed by dilation -
	mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
	# steps together are useful for removing ________ (?)

	### STEP 6 - find the biggest strawberry
	# will wrap all strawberries into ellipses and select the biggest ellipse
	big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)

	### STEP 7 - overlay the mask we created on the image
	overlay = overlay_mask(mask_clean, image)

	### STEP 8 - circle the biggest strawberry
	circled = circle_contour(overlay, big_strawberry_contour)

	show(circled)

	### STEP 9 - convert back to original color scheme
	bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)



# read the image
image = cv2.imread('berry.jpg')
result = find_strawberry(image)
# rewrite the new image
cv2.imwrite('berry2.jpg', result)
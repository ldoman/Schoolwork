"""
B457 Assignment 0
"""

__author__ = "Luke Doman"

# Imports
import time
from PIL import Image
from pylab import *
import numpy
from scipy.ndimage import filters
from matplotlib import *

# Problem 0
def p0():
	# 0.1
	im_empire = Image.open('empire.jpg')

	# 0.2
	empire_cropped = im_empire.crop((100, 100, 400, 400))
	empire_cropped.save("empire_cropped.jpg")

	empire_cropped_rot = empire_cropped.rotate(45)
	empire_cropped_rot.save("empire_cropped_rotated.jpg")

	# 0.3
	ar_emp = array(im_empire)
	ar_emp_crop = array(empire_cropped)
	ar_emp_crop_rot = array(empire_cropped_rot)

	"""
	imshow(ar_emp)
	show()

	imshow(ar_emp_crop)
	show()

	imshow(ar_emp_crop_rot)
	show()
	"""
	
	# 0.4
	empire_gray = im_empire.convert('L')
	empire_gray.save("empire_gray.jpg")

	# 0.5
	ar_emp_gray = array(empire_gray)
	ar_emp_new = toGray(im_empire)

	"""
	imshow(ar_emp_gray, cmap='gray')
	show()	
	imshow(ar_emp_new, cmap='gray')
	show()
	"""

def p1():
	# 1.1 - Please see function  convolve

	#1.2

	# Define filters
	impulse_filter = [[0,0,0],[0,1,0],[0,0,0]]
	box_filter = [[.111,.111,.111],[.111,.111,.111],[.111,.111,.111]]
	gaus_filter = [[.062,.125,.062],[.125,.25,.125],[.062,.125,.062]]

	# Load image	
	im_empire = Image.open('empire_gray.jpg')

	# Apply filters to image
	time1 = time.time()
	im_empire_impulse = convolve(im_empire, impulse_filter)
	print "--- Time 1: %s seconds ---" % (time.time() - time1)
	im_empire_box = convolve(im_empire, box_filter)
	im_empire_guas = convolve(im_empire, gaus_filter)

	# Show the new images with filters
	imshow(im_empire_impulse, cmap='gray')
	show()

	imshow(im_empire_box, cmap='gray')
	show()

	imshow(im_empire_guas, cmap='gray')
	show()

	# Convert to type image TODO something is wrong with my version of PIL and this doesn't work
	im_empire_impulse = Image.fromarray(uint8(im_empire_impulse))
	im_empire_box = Image.fromarray(im_empire_box)
	im_empire_guas = Image.fromarray(im_empire_guas)

	# Save images - can't do this since above code doesn't work...
	#im_empire_impulse.convert('RGB')	
	im_empire_impulse.save("empire_filter1.jpg", "JPEG")
	#im_empire_box.save("empire_filter2.jpg")
	#im_empire_guas.save("empire_filter3.jpg")

	sep_filter1 = [[1],[2],[1]] 
	sep_filter2 = [[1,2,1],[],[]] 

	# TODO: Update convolve
	"""
	time2 = time.time()
	im_empire_sep = convolve(im_empire, sep_filter1)
	im_empire_sep = convolve(im_empire_sep, sep_filter2)
	print "--- %s seconds ---" % (time.time() - time2)
	"""

	#1.4
	sigma = 1 #standard deviation
	ar_im_1 = array(im_empire)
	filters.gaussian_filter(im_empire, (sigma,sigma), (0,1), ar_im_1)

	sigma = 5 #standard deviation
	ar_im_2 = array(im_empire)
	filters.gaussian_filter(im_empire, (sigma,sigma), (0,1), ar_im_2)

	sigma = 10 #standard deviation
	ar_im_3 = array(im_empire)
	filters.gaussian_filter(im_empire, (sigma,sigma), (0,1), ar_im_3)

	# Show the new images with gaussian filters
	imshow(ar_im_1, cmap='gray')
	show()

	imshow(ar_im_2, cmap='gray')
	show()

	imshow(ar_im_3, cmap='gray')
	show()

	#1.5 TODO



# TODO: Fix so it can take vectors for filters
def convolve(im, fil):
	"""
	Takes two numpy arrays and applies convolution to them.
	
	Args:
		im (numpy array): Image to apply filter to 
		fil (numpy array): Filter to apply to image

	Returns:
		Numpy array that is the image with the apllied filter
	"""
	ar_im = array(im)
	im_lenx = len(ar_im)
	im_leny = len(ar_im[0])

	ar_ret = numpy.zeros(shape=(im_lenx,im_leny))

	# Iterate over each pixel of image
	for i in range(0,im_lenx):
		for j in range(0,im_leny):
			new_pixel_value = 0
			for x in range(0,len(fil)):
				for y in range(0,len(fil[0])):
					new_pixel_value = new_pixel_value + (fil[x][y]*ar_im[i-x][j-y])

			ar_ret[i][j] = new_pixel_value

	return ar_ret
	
def toGray(im):
	"""
	Take a PIL image and converts it to grayscale.

	Args:
		im (PIL Image): Image to convert

	Returns:
		Numpy array with the same xy dimensions of im
	"""
	ar = array(im)
	ar_lenx = len(ar)
	ar_leny = len(ar[0])
	ar_im = numpy.zeros(shape=(ar_lenx,ar_leny))

	# Iterate over array
	for i in range(0,ar_lenx):
		for j in range(0,ar_leny):
			# Set gray value to 0
			gray = 0

			# Incremete gray value with RGB values
			for k in range(0,3):
				gray = gray + ar[i][j][k]
			
			# Set ij index of new ar to gray/3
			ar_im[i][j] = gray/3

	return ar_im


#pil_im.show()
if __name__ == '__main__':
	#print Image.VERSION
	#p0()
	p1()

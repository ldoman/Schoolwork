"""
B457 Assignment 2
"""

__author__ = "Luke Doman"

# Imports
from matplotlib import *
import numpy as np
from PIL import Image
from pylab import *
from scipy.ndimage import filters
from scipy.signal import convolve2d

# Problem 1.1
def p1(im):
	ar_im = array(im)

	grad_1 = [-1,1]
	filter_base = np.zeros(shape=(25,25))
	imx = np.zeros(ar_im.shape)
	sigma = 2
	gaus_1 = filters.gaussian_filter(ar_im, (sigma,sigma), (0,1), imx)
	#new_filter = convolve2d(grad_1, gaus_1, mode='same')
	#gaus_1.dtype = np.uint8
	imshow(gaus_1, cmap='gray')
	show()	

def p2(im):
	ar_im = array(im)
	results = []
	results_img = []
	for f in filters:
		result = convolve2d(im, f)
		results.append(result)
		results_img.append(arr2im(result))

def euclidean_distance(v1, v2):
"""
Calculates euclidean distance between (2) 8-D vectors

Args:
	v1 (vector): Vector of 
"""

def p3(im):
	ar_im = array(im)
	

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
	

if __name__ == '__main__':
	im = Image.open('zebra.jpg').convert('L')
	p1(im)



#takes an 2D numoy array and converts it to an image
def arr2im( data ):
      data = (data - data.min())
      data *= 255/(data.max() - data.min()) 
      img = Image.fromarray(data, 'L') #the image will be saved in grayscale mode "L"
      return img
"""
img = arr2im(data)
img.save('my.png')
img.show()



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

	imshow(ar_emp)
	show()
	imshow(ar_emp_crop)
	show()
	imshow(ar_emp_crop_rot)
	show()
	
	# 0.4
	empire_gray = im_empire.convert('L')
	empire_gray.save("empire_gray.jpg")

	# 0.5
	ar_emp_gray = array(empire_gray)
	ar_emp_new = toGray(im_empire)

	imshow(ar_emp_gray, cmap='gray')
	show()	
	imshow(ar_emp_new, cmap='gray')
	show()

"""



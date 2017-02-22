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

def generate_filter(sigma, gradient, display = False):
	"""
	Generates and shows image kernel with specified parameters.

	Args:
		sigma (int): Sigma to be used for applying Gaussian filter
		gradient (2d array): Gradient to convolve Gaussian filter with
		display (bool): Whether or not to display new filter

	Returns:
		Generated filter
	"""
	# Create base filter - all black with center white pixel
	filter_base = np.zeros(shape=(51,51))
	center_pixel = int(len(filter_base)/2)
	filter_base[center_pixel][center_pixel] = 255

	gaus = filters.gaussian_filter(filter_base, (sigma,sigma))
	new_filter = convolve2d(gaus, gradient, mode='same')

	if display:
		imshow(new_filter, cmap='gray')
		show()

	return new_filter

def filter_diff(f1, f2, display = False):
	"""
	Generates a filter based on the differences between two filters.

	Args:
		f1 (filter -> 2d array): Filter 1
		f2 (filter -> 2d array): Filter 2
		display (bool): Whether or not to display new filter

	Returns:
		New filter (2d array)
	"""
	size_x = len(f1)
	size_y = len(f1[0])

	new_filter = np.zeros(shape=(size_x,size_y))

	for i in range(0, size_y):
		for j in range(0, size_x):
			new_filter[i][j] = 255 - abs(f1[i][j] - f2[i][j])

	if display:
		imshow(new_filter, cmap='gray')
		show()

	return new_filter

def array_to_image(arr):
	"""
	Takes an 2D numpy array and converts it to an image. 

	Args:
		arr (numpy 2d array): Array to convert

	Returns:
		Image
	"""
	data = (arr - arr.min())
	data *= 255/(data.max() - data.min()) 
	img = Image.fromarray(data, 'L')
	return img

# Problem 1.1 - generate all the filters
def p1(display = False):
	kernels = []
	sigmas = [2,4,8]
	gradients = [[[-1,1]],[[-1,0],[1,0]]]

	# Derivative of Gaussian filters
	for sig in sigmas:
		for grad in gradients:
			kernels.append(generate_filter(sig, grad, display))

	# Center surround filters
	gaussians = []
	for sig in sigmas:
		gaussians.append(generate_filter(sig, [[1]]))

	kernels.append(filter_diff(gaussians[0], gaussians[1], display))
	kernels.append(filter_diff(gaussians[1], gaussians[2], display))

	return kernels

# Problem 1.2 apply convolution filters to zebra image
def p2(im, display = False):
	ar_im = array(im)
	kernels = p1(display)
	results = []

	for k in kernels:
		result = convolve2d(ar_im, k, mode = 'same')
		results.append(result)

	if display:
		for res in results:
			imshow(res)
			show()

def euclidean_distance(v1, v2):
	"""
	Calculates euclidean distance between (2) 8-D vectors

	Args:
		v1 (vector): Vector of 
	"""
	# TODO

def p3(im):
	ar_im = array(im)
	
	

if __name__ == '__main__':
	im = Image.open('zebra.jpg').convert('L')
	p2(im, True)


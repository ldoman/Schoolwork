"""
B457 Assignment 3
"""

__author__ = "Luke Doman"

# Imports
import math
from matplotlib import *
import numpy as np
from PIL import Image
from pprint import pprint
from pylab import *
import random
from scipy.cluster.vq import *
from scipy.ndimage import filters
from scipy.signal import convolve2d

def generate_cluster_dict(ar_im, pixels, clusters):
	"""
	Generates a dictionary where keys are tuples in form of (row, col) 
	and values are the cluster that pixel belongs to

	Args:
		ar_im (2d array): Array representation of image
		pixels (2d array): Array containing subarrays of each pixels RGB values
		clusters (np.array): Output of vq(kmeans)

	Returns:
		Generated dict
	"""
	size_y = len(ar_im)
	size_x = len(ar_im[0])
	cluster_dict = {}
	pixel_dict = {}

	i_0 = where(clusters==0)[0]
	i_1 = where(clusters==1)[0]
	i_2 = where(clusters==2)[0]

	c_list = [i_0, i_1, i_2]
	pixels = pixels.astype(np.int)

	for c in range(0, len(c_list)):
		for i in c_list[c]:
			pixel_dict[tuple(pixels[i])] = c

	for i in range (0,size_y):
		for j in range (0,size_x):
			try:
				c_id = pixel_dict[tuple(ar_im[i][j])]
			except KeyError:
				c_id = -1
			cluster_dict[(i,j)] = c_id

	return cluster_dict

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

def get_vector(x, y, imgs):
	"""
	Get the 8-D vector of activation filters for the specified pixel of image

	Args:
		x (int): X position
		y (int): Y position
		imgs (Array of images): Activation images to generate vector from

	Returns:
		Vector of pixel values of each activation image
	"""
	v = []
	for im in imgs:
		v.append(im[x][y])

	return v

# Problem 1
def p1():
	class1 = 2.0 * randn(100,2)
	class2 = randn(100,2) + array([5,5])
	class3 = randn(50,2) + array([5,0])
	features = vstack((class1,class2,class3))
	centroids,variance = kmeans(features,3)
	code,distance = vq(features,centroids)

	figure()
	ndx = where(code==0)[0]
	plot(features[ndx,0],features[ndx,1],'*')
	ndx = where(code==1)[0]
	plot(features[ndx,0],features[ndx,1],'r.')
	ndx = where(code==2)[0]
	plot(features[ndx,0],features[ndx,1],'g.')
	plot(centroids[:,0],centroids[:,1],'go')
	axis('off')
	show()

# Problem 2
def p2(image, k = 4):
	im = array(Image.open(image).convert('RGB'))
	im = im.tolist()
	size_y = len(im)
	size_x = len(im[0])

	pixels = []
	for i in range (0,size_y):
		for j in range (0,size_x):
			pixels.append(im[i][j])

	features = array(pixels, dtype = float)
	centroids,variance = kmeans(features,k)
	code,distance = vq(features,centroids)

	figure()
	ndx = where(code==0)[0]
	plot(features[ndx,0],features[ndx,1],features[ndx,2],'*')
	ndx = where(code==1)[0]
	plot(features[ndx,0],features[ndx,1],features[ndx,2],'r.')
	ndx = where(code==2)[0]
	plot(features[ndx,0],features[ndx,1],features[ndx,2],'g.')
	plot(centroids[:,0],centroids[:,1],'go')
	axis('off')
	show()

	cluster_id = generate_cluster_dict(im, features, code)
	im2 = array(Image.open(image).convert('RGB'))
	im2 = im2.tolist()
	colors = map(matplotlib.colors.hex2color,random.sample(list(matplotlib.colors.cnames.values()),k))
	for row in range (0,size_y):
		for col in range (0,size_x):
			im2[row][col] = [x * 255 for x in colors[cluster_id[(row,col)]]]
	im = array(im2).astype(np.uint8)
	im = Image.fromarray(im,'RGB')

	imshow(im, cmap='gray')
	show()

# Problem 3
def p3(image, k = 4):
	im = Image.open(image).convert('L')
	act_imgs = a2_p2(im)
	im = array(Image.open(image).convert('RGB'))
	size_y = len(im)
	size_x = len(im[0])

	# Make im with 8d vectors for each pixel
	v_8d = []
	for i in range(0, size_y):
		for j in range(0, size_x):
			v = get_vector(i, j, act_imgs)
			v_8d.append(v)

	features = array(v_8d, dtype = float)
	centroids,variance = kmeans(features,k)
	code,distance = vq(features,centroids)

	shapes = ['*','r.','g.','b.','p.','o.']
	figure()
	for i in range(0, k):
		ndx = where(code==i)[0]
		plot(features[ndx,0],features[ndx,1],features[ndx,2],shapes[i])
	plot(centroids[:,0],centroids[:,1],'go')
	axis('off')
	show()


# Problem 1.1 - generate all the filters
def a2_p1(display = False):
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
def a2_p2(im, display = False):
	ar_im = array(im)
	kernels = a2_p1(display)
	results = []

	for k in kernels:
		result = convolve2d(ar_im, k, mode = 'same')
		results.append(result)

	if display:
		for res in results:
			imshow(res)
			show()

	return results

if __name__ == '__main__':
	#im = Image.open('zebra.jpg').convert('L')
	#p1()
	#p2('fish.jpg')
	p3('zebra.jpg')


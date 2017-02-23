"""
B457 Assignment 3
"""

__author__ = "Luke Doman"

# Imports
import math
from matplotlib import *
import numpy as np
from PIL import Image
from pylab import *
import random
from scipy.cluster.vq import *
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
def p2(image):
	# coloring the pixels
	im = array(Image.open(image).convert('RGB'))
	im = im.tolist()
	size_y = len(im)
	size_x = len(im)

	#colors = map(matplotlib.colors.hex2color,random.sample(list(matplotlib.colors.cnames.values()),3))
	pixels = []

	for i in range (0,size_y):
		for j in range (0,size_x):
			pixels.append(im[i][j])

	features = array(pixels)
	features.dtype = float

	centroids,variance = kmeans(features,4)
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


			#im2[row][col] = [x * 255 for x in colors[clust_id[row,col]]]
	#im = array(im2).astype(np.uint8)
	#im = Image.fromarray(im,'RGB')

	#imshow(im, cmap='gray')
	#show()

# Problem 1.3
def p3():
	print 1# TODO

if __name__ == '__main__':
	#im = Image.open('zebra.jpg').convert('L')
	#p1()
	p2('zebra.jpg')


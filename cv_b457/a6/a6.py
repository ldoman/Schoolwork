"""
B457 Assignment 6
"""

__author__ = "Luke Doman"

# Imports
import json
from math import sqrt
from matplotlib import *
import matplotlib.cm as cmx
import matplotlib.colors as colors
import numpy as np
import os
from PIL import Image
from pprint import pprint
from pylab import *
import random
import scipy.ndimage as ndi
from scipy.cluster.vq import *
from skimage import feature
import cv2

data_dirs = ['airplanes','camera','chair','crab','crocodile','elephant','headphone','pizza','soccer_ball','starfish']

# P1.1 & P2.0
def get_features(im, display = False):
	"""
	Take an image and runs the opencv orb function to find the features

	Args:
		img (cv2 image): Image to run on

	Returns:
		Vector of vectors for features
	"""
	# Initiate STAR detector
	orb = cv2.ORB()

	# find the keypoints with ORB
	kp = orb.detect(im,None)

	# compute the descriptors with ORB
	kp, des = orb.compute(im, kp)
	#pts = [np.array([idx.pt[0],idx.pt[1]]) for idx in kp]
	pts = np.array([idx.pt for idx in kp])
	pts = pts.reshape((-1,1,2)).astype('float32')
	if display:
		img2 = cv2.drawKeypoints(im,kp,color=(0,255,0), flags=0)
		plt.imshow(img2),plt.show()

	return kp, des, pts

# P1.2
def euclidean_dist(f1, f2):
	"""
	Calculate Euclidean distance between 2 n-dimensional SIFT features

	Args:
		f1 (Vector): SIFT feature 1
		f2 (Vector): SIFT feature 2

	Returns:
		Int
	"""
	if len(f1) != len(f2):
		print "SIFT features of different dimensionality. Aborting..."

	dist = 0
	for i in range(0, len(f1)):
		dist = dist + math.pow((f1[i]-f2[i]), 2)

	dist = sqrt(dist)
	return dist

# P1.3
def generate_dist_matrix(fv1, fv2):
	"""
	Generate dist matrix for feature vectors of 2 images

	Args:
		fv1(Vector): Vector of features form im1
		f2(Vector): Vector of features form im2

	Returns:
		2-D Matrix of Euclidean dist of each feature to another
	"""
	size_fv1 = len(fv1)
	size_fv2 = len(fv2)
	mat = np.zeros(shape=(size_fv1, size_fv2), dtype = int)

	for i in range(0, size_fv1):
		for j in range(0, size_fv2):
			dist = euclidean_dist(fv1[i],fv2[j])
			mat[i][j] = dist
			#print dist

	return mat

# P1.4
def find_matching_features(mat, thres):
	"""
	Finds features in both images whose distance is less
	than specified threshold.

	Args:
		mat (Matrix): Matrix where each value is the ith and jth features distance from another
		thres (int): Threshold used to find matches

	Returns:
		List of matches as tuples in form of (ith feature, jth feature)
	"""
	matches = []

	for i in range(0, len(mat)):
		for j in range(0, len(mat[0])):
			if mat[i][j] < thres:
				matches.append((i, j))

	return matches

# P1.5
def bf_matcher(kp1, kp2, des1, des2):
	"""
	OpenCV's feature matcher
	"""
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match descriptors.
	matches = bf.match(des1,des2)

	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)

	# Draw first 10 matches.
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

	plt.imshow(img3),plt.show()

# P2.1
def homography(src_pts, dst_pts):
	"""
	Uses OpenCV's findHomography function to find homography
	between SIFT features of 2 images.
	"""
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	return M, mask

# P2.2
def display_homography(M, mask, img1, img2):
	"""
	Visualize matching features found with homography. Code from provided link.
	"""
	img1 = np.array(img1)
	img2 = np.array(img2)
	matchesMask = mask.ravel().tolist()

	h,w = img1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)

	img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.CV_AA)
	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
		               singlePointColor = None,
		               matchesMask = matchesMask, # draw only inliers
		               flags = 2)

	img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)# Errors

	plt.imshow(img3, 'gray'),plt.show()


def get_cmap(N):
    """
	Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color. From: http://stackoverflow.com/a/25628397
	"""
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

# Problem 1.2
def bag_of_words(im, k = 4, display = False):
	"""
	Clusters SIFT features of image into k clusters and visualizes them.

	Args:
		im (CV2 image): Image to get features of
		k (int): Number of clusters
		display (bool): Whether to show plots or not

	Returns:
		List of k clusters and coordinates of the points
	"""
	kp, fv, coord = get_features(im, display)

	features = array(fv, dtype = float)
	centroids,variance = kmeans(features,k)
	code,distance = vq(features,centroids)

	shapes = ['*','r.','g.','b.','p.','o']
	figure()
	for i in range(0, k):
		ndx = where(code==i)[0]
		x_points = coord[ndx,0][:,0]
		y_points = coord[ndx,0][:,1]
		plot(x_points, y_points, shapes[len(shapes)%(i+1)-1])

	if display:
		axis('off')
		imshow(im)
		show()

	return code, coord[:,0]

def appendimages(im1,im2):
	# select the image with the fewest rows and fill in enough empty rows
	im1 = np.array(im1)
	im2 = np.array(im2)
	rows1 = im1.shape[0]
	rows2 = im2.shape[0]
	if rows1 < rows2:
		im1 = np.concatenate((im1,np.zeros((rows2-rows1,im1.shape[1]))),axis=0)
	elif rows1 > rows2:
		im2 = np.concatenate((im2,np.zeros((rows1-rows2,im2.shape[1]))),axis=0)
	# if none of these cases they are equal, no filling needed.
	return np.concatenate((im1,im2), axis=1)

# Problem 1.3 - Code given by prof
def plot_matches(im1,im2,locs1,locs2):
	im1 = np.array(im1)
	im2 = np.array(im2)
	im3 = appendimages(im1,im2)
	addr = im1.shape[0]
	addc = im1.shape[1]#+90
	plt.imshow(im3)
	cols1 = im1.shape[1]

	for i in range(len(locs1)):
		rows = [locs1[i][0], locs2[i][0]]
		cos = [locs1[i][1], locs2[i][1]+addc]
		plt.plot(cos, rows, 'k-', lw=1)
	show()

# Problem 1 execution
def p1():
	# P1.0-2
	im1 = cv2.imread('cluttered_desk.png')
	img1 = Image.open('cluttered_desk.png').convert('L')
	clust1, coord1 = bag_of_words(im1, 20)#, True)

	im2 = cv2.imread('elephant_model.png')
	img2 = Image.open('elephant_model.png').convert('L')
	clust2, coord2 = bag_of_words(im2, 20)

	im3 = cv2.imread('staple_remover_model.png')
	img3 = Image.open('staple_remover_model.png').convert('L')
	clust3, coord3 = bag_of_words(im3, 20)

	# P 1.3
	plot_matches(img1, img2, coord1, coord2)
	plot_matches(img1, img3, coord1, coord3)

	# P 1.4 TODO
	M, mask = homography(coord1, coord2)
	#display_homography(M, mask, img1, img2)# My CV2 is missing draw matches?
	#print M

# Problem 2.1
def extract_all(dirs = data_dirs):
	"""
	Extracts SIFT features from every image in our data directory.

	Args:
		dirs (List of strings): Every subdir we wish to explore

	Returns:
		100-D list of features for each image 
	"""
	features = []
	for d in dirs:
		dir_path = os.path.join(os.getcwd(), 'Data', d)
		for fp in os.listdir(dir_path):
			im = cv2.imread('cluttered_desk.png')
			kp, des, pts = get_features(im)
			features.append([kp,des,pts])

	return features

def find_centers(sift_list, k = 200):
	for i in range(0,100,10):
		fv = sift_list[i][1]#array(fv, dtype = float)
		features = array(fv, dtype = float)
		centroids,variance = kmeans(features,k)
		code,distance = vq(features,centroids)
		print len(centroids)

# Problem 2 execution
def p2():
	features = extract_all()
	find_centers(features)
	


if __name__ == '__main__':
	#p1()
	p2()




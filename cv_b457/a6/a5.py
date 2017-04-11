"""
B457 Assignment 5
"""

__author__ = "Luke Doman"

# Imports
from math import sqrt
from matplotlib import *
import numpy as np
from PIL import Image
from pprint import pprint
from pylab import *
import random
import scipy.ndimage as ndi
from skimage import feature
import cv2

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
	print M
	#print mask

# P2.2
def display_homography(M, mask, img1, img2):
	"""
	Visualize matching features found with homography. Code from provided link.
	"""
	matchesMask = mask.ravel().tolist()

	h,w = img1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)

	img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
		               singlePointColor = None,
		               matchesMask = matchesMask, # draw only inliers
		               flags = 2)

	img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

	plt.imshow(img3, 'gray'),plt.show()

if __name__ == '__main__':
	img1 = cv2.imread('box.png')
	kp1, fv1, coord1 = get_features(img1, True)
	img2 = cv2.imread('box_in_scene.png')
	kp2, fv2, coord2 = get_features(img2, True)

	dist_mat = generate_dist_matrix(fv1, fv2)
	
	matches = find_matching_features(dist_mat, 550)

	bf_matcher(kp1, kp2, fv1, fv2)

	#M, mask = homography(coord1, coord2) # Can't get the types correct

	#display_homography(M, mask, img1, img2)

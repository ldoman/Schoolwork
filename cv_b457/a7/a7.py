"""
B457 Assignment 7
"""

__author__ = "Luke Doman"

# Imports
import cv2
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
from sys import maxint

json_cache = 'a7_cache.json'
output_file = 'a7_results.txt'
data_dirs = ['airplanes','camera','chair','crab','crocodile','elephant','headphone','pizza','soccer_ball','starfish']

# Useful A5 items
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

def euclidean_dist(f1, f2):
	"""
	Calculate Euclidean distance between 2 n-dimensional SIFT features

	Args:
		f1 (Vector): SIFT feature 1
		f2 (Vector): SIFT feature 2

	Returns:
		Int
	"""
	#if len(f1) != len(f2):
		#print "SIFT features of different dimensionality. %d vs %d" % (len(f1),len(f2))

	dist = 0
	for i in range(0, len(f1)):
		dist = dist + math.pow((f1[i]-f2[i]), 2)

	dist = sqrt(dist)
	return dist

# Problem 1.1
def extract_all(dirs = data_dirs, json_file = None):
	"""
	Extracts SIFT features from every image in our data directory.
	Can load features previously extracted from a json file.

	Args:
		dirs (List of strings): Every subdir we wish to explore
		json_file (string): Name of json file to get features from
		generate_json (bool): Whether or not to genrate the json file

	Returns:
		100-D list of features for each image 
	"""
	if json_file:
		with open(os.path.join(os.getcwd(), json_file), 'r') as f:
			json_dict = json.load(f)
			features = np.array(json_dict['features'])
	else:
		features = []
		for d in dirs:
			dir_path = os.path.join(os.getcwd(), 'Data', d)
			count = 0
			for fp in os.listdir(dir_path):
				if count > 32:
					break
				im = cv2.imread(os.path.join(dir_path, fp))
				kp, des, pts = get_features(im)
				features.append([kp,des,pts])
				count += 1

	return features

# Problem 1.2
def find_centers(sift_list, k = 200, json_file = None):
	"""
	Add features from the  first of each different image type to an 
	array to perfrom k-means on. Note: K-means doesn't return exactly
	k clusters at high values.
	"""
	if json_file:
		with open(os.path.join(os.getcwd(), json_file), 'r') as f:
			json_dict = json.load(f)
			centroids = np.array(json_dict['centroids'])
	else:
		fv = []
		for i in range(0,100,10):
			for feature in sift_list[i][1]:
				fv.append(feature)
		features = array(fv, dtype = float)
		centroids,variance = kmeans(features,k)
		code,distance = vq(features,centroids)

	return centroids

# Problem 1.3
def generate_hist(features, centers, json_file = None):
	"""
	Iterates over  every feature of every image and places 
	in closest cluster center.

	Args:
		features (3d list): List of each image's features
		centers (2d list): List of calculated cluster centers
		json_file (string): Name of json file to get data from

	Returns:
		List of each image's histogram of features
	"""
	if json_file:
		with open(os.path.join(os.getcwd(), json_file), 'r') as f:
			json_dict = json.load(f)
			hists = json_dict['histograms']
			features = json_dict['features']
			centers = json_dict['centroids']

	else:
		fv_len = len(features)
		cen_len = len(centers)
		hists = [[0 for f in range(cen_len)] for im in range(fv_len)]
		for i in range(fv_len):
			im = features[i]
			for feature in im[1]:
				min_index = -1
				min_dist = maxint
				for j in range(cen_len):
					dist = euclidean_dist(feature, centers[j])
					min_index = j if dist < min_dist else min_index
					min_dist = dist if dist < min_dist else min_dist
				hists[i][min_index] += 1

	return hists

# Problem 1.4
def im_query(im, centers, im_map, json_file = json_cache):
	"""
	Finds the best 5 matches for the given image.
	"""
	# Open json file to get all histograms
	with open(os.path.join(os.getcwd(), json_file), 'r') as f:
		json_dict = json.load(f)
		hists = json_dict['histograms']
		centers = json_dict['centroids']

	# Calculate hist for passed image
	im_hist = [0 for f  in range(len(centers))]
	for feature in im:
		min_index = -1
		min_dist = maxint
		for j in range(len(centers)):
			dist = euclidean_dist(feature, centers[j])
			min_index = j if dist < min_dist else min_index
			min_dist = dist if dist < min_dist else min_dist
		im_hist[min_index] += 1

	# Get dist between passed im and every other one
	dists = []
	for i in range(len(hists)):
		dists.append((i, euclidean_dist(im_hist, hists[i])))
	dists.sort(key=lambda tup: tup[1])

	ret = [im_map[t[0]] for t in dists[:5]]
	return ret

def im_hist_map(dirs = data_dirs):
	"""
	Generates map between index of histogram nad filename.
	"""
	fnames = []
	for d in dirs:
		dir_path = os.path.join(os.getcwd(), 'Data', d)
		count = 0
		for fp in os.listdir(dir_path):
			if count > 32:
				break
			fnames.append(os.path.join('Data', d, fp))

	return fnames

def generate_json(features, centroids, hists, json_file = json_cache):
	"""
	Generates a json file with all the calculated data saved there.
	"""
	# Make vars serializable
	f = []
	for im in features:
		f.append(im[1].tolist())
	c = centroids.tolist()

	json_dict = {'features': f,
				 'centroids': c,
				 'histograms': hists}

	with open(os.path.join(os.getcwd(), json_file), 'w') as f:
		json.dump(json_dict, f)

# Problem 1 execution
def p1():
	im_map = im_hist_map()

	# P 1.1
	#features = extract_all()

	# P 1.2
	#centroids = find_centers(features)

	# P 1.3
	#hists = generate_hist(features, centroids)
	hists = generate_hist(None, None, json_cache) # Cache version
	#generate_json(features, centroids, hists) # NOTE: Only run once

	# P 1.4 
	with open(output_file, 'w') as out:
		for d in data_dirs:
			count = 0
			for fp in os.listdir(os.path.join(os.getcwd(), 'Data', d)):
				count += 1
				if count < 32:
					continue
				if count > 33:# Test
					break
				print count
				fpath = os.path.join(os.getcwd(), 'Data', d, fp)
				#out.write('')# New line
				out.write('\n\nImage matches for: %s/%s' % (d,fp))
				im = cv2.imread(fpath)
				imf = get_features(im)[1]
				#im_query(imf, centroids, im_map)
				matches = im_query(imf, None, im_map) # Cache version
				out.write(str(matches))


if __name__ == '__main__':
	p1()


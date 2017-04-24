"""
B455 Assignment 4
"""

__author__ = "Luke Doman"

# Imports
import csv
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


# Constants
data_csv = 'skylake_oc_results.csv'
test_cpus = ['6600', '6600k', '6700', '6700k']
mb_brands = ['asus', 'asrock', 'evga', 'gigabyte', 'msi']
mb_map = {'asus': 0, 'asrock': 1, 'evga': 2, 'gigabyte': 3, 'msi': 4, 'na': 5}
TRAIN_SIZE = 40
json_cache = 'a7_cache.json'
output_file = 'a7_results.txt'

def euclidean_dist(f1, f2):
	"""
	Calculate Euclidean distance between 2 n-dimensional SIFT features

	Args:
		f1 (Vector): Feature 1
		f2 (Vector): Feature 2

	Returns:
		Float
	"""
	#if len(f1) != len(f2):
		#print "SIFT features of different dimensionality. %d vs %d" % (len(f1),len(f2))

	dist = 0
	for i in range(0, len(f1)):
		dist = dist + math.pow((f1[i]-f2[i]), 2)

	dist = sqrt(dist)
	return dist

def find_centers(features, k = 30):
	"""
	Add features from the  first of each different image type to an 
	array to perfrom k-means on. Note: K-means doesn't return exactly
	k clusters at high values.
	"""
	rand_ndx = np.random.rand(k)
	fv = [features[int(v * len(features))] for v in rand_ndx]
	#fv = [int(v * len(features)) for v in rand_ndx]
	#print fv
	features = array(fv, dtype = float)
	centroids,variance = kmeans(features,k)
	code,distance = vq(features,centroids)

	return centroids

def generate_hist(features, centers):
	"""
	Iterates over  every feature of every image and places 
	in closest cluster center.

	Args:
		features (3d list): List of each image's features
		centers (2d list): List of calculated cluster centers

	Returns:
		List of each image's histogram of features
	"""
	fv_len = len(features)
	cen_len = len(centers)
	hists = [[0 for f in range(cen_len)] for im in range(fv_len)]
	for i in range(fv_len):
		min_index = -1
		min_dist = maxint
		for j in range(cen_len):
			dist = euclidean_dist(features[i], centers[j])
			min_index = j if dist < min_dist else min_index
			min_dist = dist if dist < min_dist else min_dist
		hists[i][min_index] += 1

	return hists

# Problem 1.4
def f_query(f, centers):
	"""
	Finds the best 5 matches for the given feature vector.
	"""
	# Calculate hist for passed feature
	im_hist = [0 for f  in range(len(centers))]
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
	#print "done"
	# P 1.4 
	all_matches = []
	total_count = 0
	with open(output_file, 'w') as out:
		for d in data_dirs:
			count = 0
			for fp in sorted(os.listdir(os.path.join(os.getcwd(), 'Data', d))):
				count += 1
				total_count += 1
				if count < TRAIN_SIZE:
					continue
				if count > 100:# Test
					break
				fpath = os.path.join(os.getcwd(), 'Data', d, fp)
				out.write('\n\nImage matches for: %s/%s' % (d,fp))
				im = cv2.imread(fpath)
				imf = get_features(im)[1]
				#im_query(imf, centroids, im_map)
				matches = im_query(imf, None, im_map) # Cache version
				all_matches.append((d, matches))
				out.write(str(matches))

	# Evaluate performance
	print "%d images processed" % (total_count)
	class_perf = {}
	for class_name, matches in all_matches:
		if class_name not in class_perf:
			class_perf[class_name] = (0,0)
		classes = [m.split('/')[1] for m in matches]
		class_weights = np.array([.5, .4, .3, .3, .2])
		class_values = np.array([1 if c == class_name else 0 for c in classes])
		p = np.dot(class_values.T, class_weights)
		if p > .49:
			class_perf[class_name] = (class_perf[class_name][0] + 1, class_perf[class_name][1] + 1)
		else:
			class_perf[class_name] = (class_perf[class_name][0], class_perf[class_name][1] + 1)

	for class_name, stats in class_perf.iteritems():
		print "%s classification accuracy: %d%s" % (class_name, ((float(stats[0])/float(stats[1])))*100, '%')


def parse_csv(file_path):
	"""
	Read csv, convert vals to appropriate data type, and return all features.

	Args:
		file_path (string): Path to csv to parse

	Returns:
		Numpy array of all feature arrays
	"""
	features = []
	with open(file_path, 'r') as data:
		csv_reader = csv.DictReader(data)    
		for line in csv_reader:
			if line['cpu'] not in test_cpus:
				#print "%s not in test cpus" % line['cpu']
				continue
			bclk = float(line['bclk'])
			mul = int(line['core_mult'])
			core_freq = float(line['core_freq'])
			cache_freq = float(line['cache_freq'])
			vcore = float(line['vcore'])
			fclk = float(line['fclk']) if line['fclk'] else 0# TODO
			bat = 0#(line[''])
			ram = int(line['ram'].split()[0]) if line['ram'] else 0# TODO
			mb_ = line['mb'].split()[0].lower() if line['mb'].split()[0].lower() in mb_brands else 'na'
			mb = mb_map[mb_]
			arr = np.array([bclk, mul, core_freq, cache_freq, vcore, fclk, bat, ram, mb])
			features.append(arr)

	return np.array(features)










if __name__ == '__main__':
	f = parse_csv(data_csv)
	centers = find_centers(f)
	print centers




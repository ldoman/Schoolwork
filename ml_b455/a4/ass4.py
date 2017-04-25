"""
B455 Assignment 4
"""

__author__ = "Luke Doman"

# Imports
import csv
from dataloader import splitdataset
from math import sqrt
#from matplotlib import *
import numpy as np
import os
from pprint import pprint
import random
import regressionalgorithms as algs
from utilities import *

# Constants
TRAIN_SIZE = 40
TEST_SIZE = 90
data_csv = 'skylake_oc_results.csv'
test_cpus = ['6600', '6600k', '6700', '6700k']
mb_map = {'asus': 0, 'asrock': 1, 'evga': 2, 'gigabyte': 3, 'msi': 4, 'na': 5}

def euclidean_dist(f1, f2):
	"""
	Calculate Euclidean distance between 2 n-dimensional SIFT features

	Args:
		f1 (Vector): Feature 1
		f2 (Vector): Feature 2

	Returns:
		Float
	"""
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

def parse_csv(file_path):
	"""
	Read csv, convert vals to appropriate data type, and return all features.

	Args:
		file_path (string): Path to csv to parse

	Returns:
		Numpy array of features (X), Numpy array of attained clockspeeds (Y)
	"""
	features = []

	with open(file_path, 'r') as data:
		csv_reader = csv.DictReader(data)   
		for line in csv_reader:
			if line['cpu'] not in test_cpus:
				continue
			bclk = float(line['bclk'])
			mul = int(line['core_mult'])
			core_freq = float(line['core_freq'])
			cache_freq = float(line['cache_freq'])
			vcore = float(line['vcore'])
			#bat = float(line['batch']) # Need much more data for this to be relevant
			fclk = float(line['fclk']) if line['fclk'] else 1 # Default speeed
			ram = int(line['ram'].split()[0]) if line['ram'] else 2000 # Mean ram speed
			mb = mb_map[line['mb'].split()[0].lower()] if line['mb'].split()[0].lower() in mb_map else len(mb_map)
			arr = np.array([bclk, mul, cache_freq, vcore, fclk, ram, mb, core_freq])
			features.append(arr)

	return np.array(features)

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1)

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]

if __name__ == '__main__':
    trainsize = 40
    testsize = 90
    numruns = 1

    regressionalgs = {'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
#                'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                'FSLinearRegression50': algs.FSLinearRegression({'features': range(8)}),
				#'FSLinearRegression75': algs.FSLinearRegression({'features': [random.randrange(0,385) for x in range(0,75)]}),
#				'FSLinearRegression385': algs.FSLinearRegression({'features': range(385)}),
                'RidgeLinearRegression': algs.RidgeLinearRegression(),
#                'Lasso': algs.LassoRegression(),
#                'BatchGradientDescent': algs.BatchGradientDescent(),
#                'StochasticGradientDescent': algs.StochasticGradientDescent(),

             }
    numalgs = len(regressionalgs)

    # Enable the best parameter to be selected, to enable comparison
    # between algorithms with their best parameter settings
    parameters = (
        {'regwgt': 0.0},
        {'regwgt': 0.01},
        {'regwgt': 1.0},
                      )
    numparams = len(parameters)
    
    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))

    for r in range(numruns):
        trainset, testset = splitdataset(parse_csv(data_csv), trainsize, testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in regressionalgs.items():
                # Reset learner for new parameters
                learner.reset(params)
                print ('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
                # Train model
                learner.learn(trainset[0], trainset[1])
                # Test model
                predictions = learner.predict(testset[0])
                error = geterror(testset[1], predictions)
                print ('Error for ' + learnername + ': ' + str(error))
                errors[learnername][p,r] = error


    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][0,:])
        errs = errors[learnername]
        #print ('Standard error for ' + learnername + ': ' + str(stdev(errs)/math.sqrt(len(errs)))) # TODO: Test
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p,:])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print ('Best parameters for ' + learnername + ': ' + str(learner.getparams()))
        print ('Average error for ' + learnername + ': ' + str(besterror))


"""
B455 Assignment 4
"""

__author__ = "Luke Doman"

# Imports
import csv
from dataloader import splitdataset
#from matplotlib import *
import numpy as np
import os
from pprint import pprint
import random
from scipy import stats
from sklearn import svm, neural_network, linear_model

# Constants

data_csv = 'skylake_oc_results.csv'
test_cpus = ['6600', '6600k', '6700', '6700k']
mb_map = {'asus': 0, 'asrock': 1, 'evga': 2, 'gigabyte': 3, 'msi': 4, 'na': 5}

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

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]

if __name__ == '__main__':
    trainsize = 55
    testsize = 95
    numruns = 50

    regressionalgs = {'SVM': svm.SVR(),
                'NN': neural_network.MLPRegressor(),
                'Linear': linear_model.LinearRegression()}
    numalgs = len(regressionalgs)

    preds = {'SVM': [],
                'NN': [],
                'Linear': []}

    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros(numruns)

    for r in range(numruns):
        trainset, testset = splitdataset(parse_csv(data_csv), trainsize, testsize)
        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))
        for learnername, learner in regressionalgs.items():
            learner.fit(trainset[0], trainset[1])
            predictions = learner.predict(testset[0])
            preds[learnername].append(predictions)
            error = geterror(testset[1], predictions)
            print ('Error for ' + learnername + ': ' + str(error))
            errors[learnername][r] = error

    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][:])
        errs = errors[learnername]
        print ('Average error for ' + learnername + ': ' + str(besterror))

    # Compare algorithm results. Using pairwise t-test from scipy based documentation at:
    # https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.ttest_rel.html
    np.random.seed(12345678)
    for learner, pred in preds.iteritems():
        for learner2, pred2 in preds.iteritems():
            if learner == learner2:
                continue
            pvals = []
            for i in range(len(pred)):
                pvals.append(stats.ttest_rel(pred[i], pred2[i])[1])
            ttest = stats.ttest_rel(pred[0], pred2[0])
            avg = np.mean(np.array(pvals))
            print "Mean p-value of similarity between %s and %s: %f" % (learner, learner2, avg)


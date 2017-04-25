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
from sklearn import svm, neural_network, linear_model

# Constants
TRAIN_SIZE = 40
TEST_SIZE = 90
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


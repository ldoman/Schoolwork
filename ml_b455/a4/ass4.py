"""
B455 Assignment 4

Dataset: https://docs.google.com/spreadsheets/d/1wQwYMGsSnMpKxrEesNVUSoP7hGykFWw1ygJsxwx64e8/edit#gid=0

For this assignment I wanted to apply machine learning techniques to learn functions that can predict
a cpu's frequency (in GHz) based on several parameters specific to overclocking it. The cpus of interest
are Intel's 6600k/6700k, both of which are the same architecture with only minor differences. Since 
splitting the dataset between the two would make for very small datasets I'm making the assumption that
they are the same. 

This is an interesting relationship to model because there is variance between individual chip's performance
due to imperfections during the fabrication process. Companies exist that use this information to sell
cpu's at prices relative to the performace of the mean. (https://siliconlottery.com/)

The algorithms I'm comparing are scikit's Linear Regression, Support Vector Machine, and Neural Net.
All of the algorithms performed quite well considering the dataset only contained 150 samples. I searched
for larger sets, but could not find any with the amount of features per sample that were ideal. It is clear
from the results (below) that the NN needed more data to converge to an optimal model. Each time the training
size increased it improved performance. I attempted to do feature scaling hoping it would aid it's performace
as the documentation recommended, but it ended up causing strange results so it remains commented out.

What is quite impressive is how little data SVM needs to find an extremely well fitting model. It
actually appears that it overfits as the training size increases which hurts its performace. Linear
performance trailed SVM in small training sets, but surpassed it's performance in larger sets. The p-values
from performing pairwise t-tests between the different models show that we cannot conclude that any of 
the models are definitively better than another. I believe if there was a much larger dataset to run 
on this may not be the case. 

If these results show anything, it is that picking the correct algorithm for the correct type and 
size of dataset is extremely important.

Results (250 runs each):

trainsize = 35
testsize = 115
Average error for SVM: 0.0133998281309
Average error for Linear: 2.60872702209
Average error for NN: 4.5088206317
Mean p-value of similarity between SVM and Linear: 0.198458
Mean p-value of similarity between SVM and NN: 0.203929
Mean p-value of similarity between Linear and NN: 0.220347

trainsize = 55
testsize = 95
Average error for SVM: 0.0145768332853
Average error for Linear: 0.976200999188
Average error for NN: 3.44930573181
Mean p-value of similarity between SVM and Linear: 0.240656
Mean p-value of similarity between SVM and NN: 0.273985
Mean p-value of similarity between Linear and NN: 0.269063

trainsize = 75
testsize = 75
Average error for SVM: 0.016342954674
Average error for Linear: 0.724208075786
Average error for NN: 2.91325582858
Mean p-value of similarity between SVM and Linear: 0.272095
Mean p-value of similarity between SVM and NN: 0.318574
Mean p-value of similarity between Linear and NN: 0.338282

trainsize = 95
testsize = 55
Average error for SVM: 0.0185912457013
Average error for Linear: 0.43114897987
Average error for NN: 1.83590584834
Mean p-value of similarity between SVM and Linear: 0.367647
Mean p-value of similarity between SVM and NN: 0.338813
Mean p-value of similarity between Linear and NN: 0.315755

trainsize = 115
testsize = 35
Average error for SVM: 0.0237080150213
Average error for Linear: 0.169883772977
Average error for NN: 0.715837655653
Mean p-value of similarity between SVM and Linear: 0.390834
Mean p-value of similarity between SVM and NN: 0.390136
Mean p-value of similarity between Linear and NN: 0.401532

"""

__author__ = "Luke Doman"

# Imports
import csv
from dataloader import splitdataset
import numpy as np
import os
from pprint import pprint
import random
from scipy import stats
from sklearn import svm, neural_network, linear_model, preprocessing

# Constants
data_csv = 'skylake_oc_results.csv'
test_cpus = ['6600', '6600k', '6700', '6700k']
mb_map = {'asus': 0, 'asrock': 1, 'evga': 2, 'gigabyte': 3, 'msi': 4, 'na': 5}
my_cpu = [100, 4, 1.35, 1, 3000, 4, 4.6]

def parse_csv(file_path):
	"""
	Read csv, convert vals to appropriate data type, and return all features.

	Args:
		file_path (string): Path to csv to parse

	Returns:
		Numpy array of features with last column being core frequency (y)
	"""
	features = []

	with open(file_path, 'r') as data:
		csv_reader = csv.DictReader(data)   
		for line in csv_reader:
			if line['cpu'] not in test_cpus:
				continue
			bclk = float(line['bclk'])
			#mul = int(line['core_mult']) # This makes it too easy to guess. core_freq = bclk * mult
			core_freq = float(line['core_freq'])
			cache_freq = float(line['cache_freq'])
			vcore = float(line['vcore'])
			# Need more data for batch # to be relevant, but it would be a very indicative feature.
			#bat = float(line['batch']) 
			fclk = float(line['fclk']) if line['fclk'] else 1 # Default speed is 1
			ram = int(line['ram'].split()[0]) if line['ram'] else 2000 # Default to mean ram speed
			mb = mb_map[line['mb'].split()[0].lower()] if line['mb'].split()[0].lower() in mb_map else len(mb_map)
			arr = np.array([bclk, cache_freq, vcore, fclk, ram, mb, core_freq])
			features.append(arr)

	return np.array(features)

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]

if __name__ == '__main__':
    trainsize = 75
    testsize = 75
    numruns = 250

    regressionalgs = {'SVM': svm.SVR(),
                'NN': neural_network.MLPRegressor(),
                'Linear': linear_model.LinearRegression()}
    numalgs = len(regressionalgs)

    preds = {'SVM': [],
                'NN': [],
                'Linear': []}

	# Predictions for just my cpu
    my_preds = {'SVM': [],
                'NN': [],
                'Linear': []}

    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros(numruns)

    for r in range(numruns):
        trainset, testset = splitdataset(parse_csv(data_csv), trainsize, testsize)

        """
        # Scale features as seen at:
		# http://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use
        scaler = preprocessing.StandardScaler()
        scaler.fit(trainset[0])
        trainset = (scaler.transform(trainset[0]), trainset[1])
        testset = (scaler.transform(testset[0]), testset[1])
        """

        print(('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],r))
        for learnername, learner in regressionalgs.items():
            learner.fit(trainset[0], trainset[1])
            predictions = learner.predict(testset[0])
            my_pred = learner.predict([my_cpu])
            my_preds[learnername].append(my_pred)
            preds[learnername].append(predictions)
            error = geterror(testset[1], predictions)
            print ('Error for ' + learnername + ': ' + str(error))
            errors[learnername][r] = error

    for learnername in regressionalgs:
        besterror = np.mean(errors[learnername][:])
        errs = errors[learnername]
        print ('Average error for ' + learnername + ': ' + str(besterror))

    # Compare algorithm results. Using pairwise t-test from scipy. Based on documentation at:
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

    """
    for learner, pred in my_preds.iteritems():
        my_pred = np.mean(pred)
        print "%s prediction for my cpu: %f" % (learner, my_pred)
    """

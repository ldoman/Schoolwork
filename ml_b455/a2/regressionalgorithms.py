from __future__ import division  # floating point division
import numpy as np
import math
import sys
import utilities as utils

class Regressor:
    """
    Generic regression interface; returns random regressor
    Random regressor randomly selects w from a Gaussian distribution
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        """ Reset learner """
        self.weights = None
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        self.weights = None
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class RangePredictor(Regressor):
    """
    Random predictor randomly selects value between max and min in training set.
    """

    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.min = 0
        self.max = 1
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.min = np.amin(ytrain)
        self.max = np.amax(ytrain)

    def predict(self, Xtest):
        ytest = np.random.rand(Xtest.shape[0])*(self.max-self.min) + self.min
        return ytest

class MeanPredictor(Regressor):
    """
    Returns the average target value observed; a reasonable baseline
    """
    def __init__( self, parameters={} ):
        self.params = {}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.mean = None
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.mean = np.mean(ytrain)

    def predict(self, Xtest):
        return np.ones((Xtest.shape[0],))*self.mean


class FSLinearRegression(Regressor):
    """
    Linear Regression with feature selection, and ridge regularization
    """
    def __init__( self, parameters={} ):
        self.params = {'features': [1,2,3,4,5]}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Dividing by numsamples before adding ridge regularization
        # to make the regularization parameter not dependent on numsamples
        numsamples = Xtrain.shape[0]
        Xless = Xtrain[:,self.params['features']]
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class RidgeLinearRegression(Regressor):
    """
    Linear Regression with ridge regularization (l2 regularization)
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain, lam = .01):
        """ Learns using the traindata """
        xtx = np.dot(Xtrain.T, Xtrain)
        lambda_i = lam * np.eye(Xtrain.shape[1])
        self.weights = np.dot(np.linalg.inv(xtx + lambda_i), np.dot(Xtrain.T, ytrain))

    def predict(self, Xtest):
        return np.dot(Xtest, self.weights)

class MPLinearRegression(Regressor):
    """
    TODO
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def select_features(self, X, y, e):
        x_tilde = X

    def learn(self, Xtrain, ytrain):
            """ Learns using the traindata """
            # Dividing by numsamples before adding ridge regularization
            # to make the regularization parameter not dependent on numsamples
            numsamples = Xtrain.shape[0]
            Xless = Xtrain[:,self.params['features']]
            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        Xless = Xtest[:,self.params['features']]
        ytest = np.dot(Xless, self.weights)
        return ytest

class Lasso(Regressor):
    """
    Problem 2e
    """
    def __init__( self, parameters={}, lam = .01 ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def select_features(self, X, y, e):
        x_tilde = X

    def err(self, X, y, w, lam):
        """
        Error calculation as defined on page 62 of notes.

        Args:
            X (Matrix): Input data
            y (Vector): Output vector
            w (float?): Weight
            lam (float): Regularization parameter

        Returns:
            Error
        """
        y = y[:, np.newaxis]
        return np.dot((np.dot(X,w) - y).T, np.dot(X,w) - y) + (lam * np.linalg.norm(w))

    def prox(self, n, lam, w):
        """
        Thresholding fuction as defined on page 62 of notes.

        Args:
            n (int): Stepsize
            lam (float): Regularization parameter
            w (float?): Weight

        Returns:
            Updated weight
        """
        nl = n * lam
        if w.any() > nl:
            return w - nl
        elif abs(w).any() <= nl:
            return 0
        else:
            return w + nl

    def bgd(self, X, y, lam):
        """
        Batch gradient descent as defined on page 62 of notes.

        Args:
            X (Matrix): Input data
            y (Vector): Output vector
            lam (float): Regularization parameter

        Returns:
            Updated weights
        """
        numsamples = X.shape[0]
        w = 0
        err = sys.maxsize
        tolerance = .18
        xx = np.dot(X.T,X)/numsamples
        xy = np.dot(X.T,y)/numsamples
        n = 1/(2 * np.linalg.norm(xx))
        iterations = 100000
        i = 0

        #print "xx: " + str(xx)
        #print "xy: " + str(xy)
        #print "n: " + str(n)

        while abs(self.err(X, y, w, lam) - err).any() > tolerance:
            if i > iterations:
                print "This is taking a while"
            err = self.err(X, y, w, lam)
            arg = w - np.dot(np.dot(n, xx), w) + np.dot(n, xy)
            w = self.prox(n, lam, arg)
            i = i + 1

        return w

    def learn(self, Xtrain, ytrain, lam = .01):
        """ Learns using the traindata """
        self.weights = self.bgd(Xtrain, ytrain, lam)

    def predict(self, Xtest):
        return np.dot(Xtest, self.weights)


class BatchGradientDescent(Regressor):
    """
    Problem 2e
    """
    def __init__( self, parameters={}, lam = .01 ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def select_features(self, X, y, e):
        x_tilde = X

    def err(self, X, y, w, lam):
        """
        Error calculation as defined on page 62 of notes.

        Args:
            X (Matrix): Input data
            y (Vector): Output vector
            w (float?): Weight
            lam (float): Regularization parameter

        Returns:
            Error
        """
        y = y[:, np.newaxis]
        return np.dot((np.dot(X,w) - y).T, np.dot(X,w) - y) + (lam * np.linalg.norm(w))

    def prox(self, n, lam, w):
        """
        Thresholding fuction as defined on page 62 of notes.

        Args:
            n (int): Stepsize
            lam (float): Regularization parameter
            w (float?): Weight

        Returns:
            Updated weight
        """
        nl = n * lam
        if w.any() > nl:
            return w - nl
        elif abs(w).any() <= nl:
            return 0
        else:
            return w + nl

    def bgd(self, X, y, lam):
        """
        Batch gradient descent as defined on page 62 of notes.

        Args:
            X (Matrix): Input data
            y (Vector): Output vector
            lam (float): Regularization parameter

        Returns:
            Updated weights
        """
        numsamples = X.shape[0]
        w = 0
        err = sys.maxsize
        tolerance = .18
        xx = np.dot(X.T,X)/numsamples
        xy = np.dot(X.T,y)/numsamples
        n = 1/(2 * np.linalg.norm(xx))
        iterations = 100000
        i = 0

        #print "xx: " + str(xx)
        #print "xy: " + str(xy)
        #print "n: " + str(n)

        while abs(self.err(X, y, w, lam) - err).any() > tolerance:
            if i > iterations:
                print "This is taking a while"
            err = self.err(X, y, w, lam)
            arg = w - n * xx * w + n * xy
            w = self.prox(n, lam, arg)
            i = i + 1

        return w

    def learn(self, Xtrain, ytrain, lam = .01):
        """ Learns using the traindata """
        self.weights = self.bgd(Xtrain, ytrain, lam)

    def predict(self, Xtest):
        return np.dot(Xtest, self.weights)

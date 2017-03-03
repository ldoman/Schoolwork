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

# TODO: Implement
class MPLinearRegression(Regressor):
    """
    Problem 2d
    """
    def __init__( self, parameters={} ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def learn(self, Xtrain, ytrain):
            """ Learns using the traindata """
            self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xless.T,Xless)/numsamples), Xless.T),ytrain)/numsamples

    def predict(self, Xtest):
        return np.dot(Xtest, self.weights)

# TODO: Runs forever.
class LassoRegression(Regressor):
    """
    Problem 2e
    """
    def __init__( self, parameters={}, lam = .01 ):
        # Default parameters, any of which can be overwritten by values passed to params
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

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

    def bgd_l1(self, X, y, lam):
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
                print "Iteration limit exceeded."
                break
            err = self.err(X, y, w, lam)
            arg = w - np.dot(np.dot(n, xx), w) + np.dot(n, xy)
            w = self.prox(n, lam, arg)
            i = i + 1

        return w

    def learn(self, Xtrain, ytrain, lam = .01):
        """ Learns using the traindata """
        self.weights = self.bgd_l1(Xtrain, ytrain, lam)

    def predict(self, Xtest):
        return np.dot(Xtest, self.weights)

# TODO: Fix
class StochasticGradientDescent(Regressor):
    """
    Problem 2f
    """
    def __init__( self, parameters={}, lam = .01 ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def err(self, X, y, t, w):
        """
        Error calculation as defined on page 62 of notes.

        Args:
            X (Matrix): Input data
            y (Vector): Output vector
            t (int): Current iteration
            w (float): Weight

        Returns:
            Error
        """
        #y = y[:, np.newaxis]
        return np.dot((np.dot(X[t].T, w) - y[t]), X[t])

    def sgd(self, X, y):
        """
        Stochastic gradient descent as defined on page 63 of notes.

        Args:
            X (Matrix): Input data
            y (Vector): Output vector

        Returns:
            Updated weights
        """
        numsamples = X.shape[0]
        w = 0
        n = .01

        for t in range(0, numsamples):
            g = self.err(X, y, t, w)
            w = w - np.dot(n, g)

        return w

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = self.sgd(Xtrain, ytrain)

    def predict(self, Xtest):
        return np.dot(Xtest, self.weights)

# TODO: Fix
class BatchGradientDescent(Regressor):
    """
    Problem 2g
    """
    def __init__( self, parameters={}, lam = .01 ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def err(self, X, y, w):
        """
        Error calculation as defined on page 62 of notes.

        Args:
            X (Matrix): Input data
            y (Vector): Output vector
            w (float): Weight

        Returns:
            Error
        """
        y = y[:, np.newaxis]
        return np.dot((np.dot(X,w) - y).T, np.dot(X,w) - y)

    def line_search(self, w, g, err):
        """
        Finds step size

        Args:
            w (float): Weight
            g (Vector): Gradient
            err (Vector): Error

        Returns:
            Step size
        """
        # TODO: Implement
        return -1


    def bgd(self, X, y, lam):
        """
        Batch gradient descent as defined on page 63 of notes.

        Args:
            X (Matrix): Input data
            y (Vector): Output vector

        Returns:
            Updated weights
        """
        numsamples = X.shape[0]
        w = 0
        err = sys.maxsize
        tolerance = .18
        xx = np.dot(X.T,X)/numsamples
        xy = np.dot(X.T,y)/numsamples
        iterations = 100000
        i = 0

        while abs(self.err(X, y, w) - err).any() > tolerance:
            if i > iterations:
                print "Iteration limit exceeded."
                break
            err = self.err(X, y, w, lam)
            g = np.dot(X.T, np.dot(X,w) - y[:, np.newaxis])/numsamples
            n = self.line_search(w, g, err)
            w = w - np.dot(g,n)
            i = i + 1

        return w

    def learn(self, Xtrain, ytrain, lam = .01):
        """ Learns using the traindata """
        self.weights = self.bgd(Xtrain, ytrain, lam)

    def predict(self, Xtest):
        return np.dot(Xtest, self.weights)

from __future__ import division  # floating point division
import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
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
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it should ignore this last feature
        self.params = {'usecolumnones': False}
        self.reset(parameters)
            
    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        self.c0_means = []
        self.c0_stds = []
        self.c1_means = []
        self.c1_stds = []

    def calc_vars(self, Xtrain, ytrain):
        """ 
        Calcuates the mean and std dev for each feature of the different classes. 
        We only need to do this once per dataset.
        """
        class0 = []
        class1 =[]
        c0m = np.zeros(len(Xtrain[0]))
        c1m = np.zeros(len(Xtrain[0]))
        c0s = np.zeros(len(Xtrain[0]))
        c1s = np.zeros(len(Xtrain[0]))
        
        for i in range(Xtrain.shape[0]):
            if ytrain[i] == 0:
                class0.append(Xtrain[i])
            else:
                class1.append(Xtrain[i])

        class0 = np.array(class0)
        class1 = np.array(class1)

        for j in range(0, len(Xtrain[0])):
            c0m[j] = np.mean(class0[:,j])
            c1m[j] = np.mean(class1[:,j])
            c0s[j] = np.std(class0[:,j])
            c1s[j] = np.std(class1[:,j])

        return c0m, c0s, c1m, c1s

    def learner_ones(self, Xtrain, ytrain):
        self.weights = np.ones(len(Xtrain[0]))
        c0_means, c0_stds, c1_means, c1_stds = self.calc_vars(Xtrain, ytrain)

        for i in range(Xtrain.shape[0]):
            for j in range(0, len(Xtrain[0])):
                mean = c0_means[j] if ytrain[i] == 0 else c1_means[j]
                std = c0_stds[j]**2 if ytrain[i] == 0 else c1_stds[j]**2
                exp = np.exp(-(np.power(Xtrain[i][j]-mean,2)/(2*std)))
                p = (1 / (np.sqrt(2 * np.pi * std))) * exp
                self.weights[j] = self.weights[j] * p

    def learner(self, Xtrain, ytrain):
        self.weights = np.ones(len(Xtrain[0]))
        self.weights[-1] = 0
        c0_means, c0_stds, c1_means, c1_stds = self.calc_vars(Xtrain, ytrain)

        for i in range(Xtrain.shape[0]):
            for j in range(0, len(Xtrain[0])-1):
                mean = c0_means[j] if ytrain[i] == 0 else c1_means[j]
                std = c0_stds[j]**2 if ytrain[i] == 0 else c1_stds[j]**2
                exp = np.exp(-(np.power(Xtrain[i][j]-mean,2)/(2*std)))
                p = (1 / (np.sqrt(2 * np.pi * std))) * exp
                self.weights[j] = self.weights[j] * p

    def learn(self, Xtrain, ytrain):
        """
        Router for calling correct learner function. Learner implementation 
        based on notes pages 77-80.
        """
        if self.params['usecolumnones'] is True:
            self.learner_ones(Xtrain, ytrain)
        else:
            self.learner(Xtrain, ytrain)

    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest

class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        self.step_size = .05 # Using fixed step size for simplicity
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    def learn(self, Xtrain, ytrain):
        """
        Logistic Regression with stochastic optimization. Based on notes
        pages 71-77, and algorithm provided on page 63.
        """
        self.weights = np.ones(len(Xtrain[0]))
        for j in range(20): # Running until convergence takes a while...
            for i in range(Xtrain.shape[0]):
                xtw = np.dot(Xtrain[i], self.weights)
                delta = np.divide((2*ytrain[i]-1)*np.sqrt(np.square(xtw)+1)-xtw,np.square(xtw)+1)
                delta = np.dot(Xtrain[i].T,delta)
                d1 = np.divide((2*ytrain[i]-1)*xtw - np.sqrt(np.square(xtw)+1)-xtw,np.power(np.square(xtw)+1,3/2))
                d2 = 2*xtw*np.divide((2*ytrain[i]-1)*np.sqrt(np.square(xtw)+1)-xtw,np.square(np.square(xtw)+1))
                hess = np.dot(Xtrain[i].T,Xtrain[i])*(d1-d2)
                self.weights = self.weights + self.step_size * delta/hess

    def predict(self, Xtest):
        # print self.weights
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest[ytest >= 0.5] = 1     
        ytest[ytest < 0.5] = 0    
        return ytest          
           

class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)        

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None
        
    # TODO: implement learn and predict functions                  

    
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)

                
           
    

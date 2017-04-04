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

    def calc_vars(self, X):
        """ Calcuates the mean and std dev for the different classes. 
        We only need ot do this once per dataset.
        """
        class0 = []
        class1 =[]
        c0m = []
        c1m = []
        c0s = []
        c1s = []
        
        for i in range(X.shape[0]):
            if X[-1] == 0:
                class0.append(X[i])
            else:
                class1.append(X[i])

        class0 = array(class0)
        class1 = array(class1)

        for j in range(0, len(Xtrain[0])):
            c0m = np.mean(class0)
            c1m = np.mean(class1)
            c0s = np.std(class0)
            c1s = np.std(class1)

        return c0m, c1m, c0s, c1s

    def learner_ones(self, Xtrain, ytrain):
        self.weights = np.ones(len(Xtrain[0]))
        self.means = np.zeros(len(Xtrain[0]))
        self.stds = np.zeros(len(Xtrain[0]))
        for j in range(0, len(Xtrain[0])):
            self.means[j] = np.mean(Xtrain[:,j])
            self.stds[j] = np.std(Xtrain[:,j])

        for i in range(Xtrain.shape[0]):
            for j in range(0, len(Xtrain[0])):
                exp = np.exp(-(np.power(Xtrain[i][j]-self.means[j],2)/(2*np.power(self.stds[j],2))))
                p = (1 / (np.sqrt(2*np.pi) * self.stds[j])) * exp
                self.weights[j] = self.weights[j] * p

    def learner(self, Xtrain, ytrain):
        self.weights = np.ones(len(Xtrain[0]))
        self.weights[-1] = 0
        self.means = np.zeros(len(Xtrain[0])-1)
        self.stds = np.zeros(len(Xtrain[0])-1)
        ps = []
        for j in range(0, len(Xtrain[0])-1):
            self.means[j] = np.mean(Xtrain[:,j])
            self.stds[j] = np.std(Xtrain[:,j])

        for i in range(Xtrain.shape[0]):
            #print self.weights
            for j in range(0, len(Xtrain[0])-1):
                exp = np.exp(-(np.power(Xtrain[i][j]-self.means[j],2)/(2*np.power(self.stds[j],2))))
                p = (1 / (np.sqrt(2*np.pi * np.power(self.stds[j], 2)))) * exp
                ps.append(p)
                self.weights[j] = self.weights[j] * p
        print '\n\n'
        print max(ps)
        print min(ps)

    def learn(self, Xtrain, ytrain):
        """
        Router for calling correct learner function
        """
        if self.params['usecolumnones'] is True:
            self.learner_ones(Xtrain, ytrain)
        else:
            self.learner(Xtrain, ytrain)

    def predict(self, Xtest):# TODO fix
        ytest = np.dot(Xtest, self.weights)
        print '\n\n'
        print self.weights
        print Xtest
        print ytest
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
        self.step_size = .05
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    def learn(self, Xtrain, ytrain):
        self.weights = np.ones(len(Xtrain[0]))
        xvec = np.dot(Xtrain, self.weights)
        p = utils.sigmoid(np.dot(Xtrain, self.weights))  #(500*9) * (9*1) = 500*1
        P = np.diagflat(p)

        for j in range(500):
            for i in range(Xtrain.shape[0]):
                xvec = np.dot(Xtrain[i], self.weights)  #(1*9) * (9*1) = 500*1
                delta = np.divide((2*ytrain[i]-1)*np.sqrt(np.square(xvec)+1)-xvec,np.square(xvec)+1)
                delta = np.dot(Xtrain[i].T,delta)
                first_term = np.divide((2*ytrain[i]-1)*xvec - np.sqrt(np.square(xvec)+1)-xvec,np.power(np.square(xvec)+1,3/2))
                second_term = 2*xvec*np.divide((2*ytrain[i]-1)*np.sqrt(np.square(xvec)+1)-xvec,np.square(np.square(xvec)+1))
                hessian = np.dot(Xtrain[i].T,Xtrain[i])*(first_term-second_term)
                self.weights = self.weights + self.step_size * delta/hessian #(500*9) * (500*1) = (9*1)
                #print self.weights


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

                
           
    

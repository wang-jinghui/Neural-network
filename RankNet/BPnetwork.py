#-*- coding:utf-8 -*-
import numpy as np
import time

def add_bias(mat):
    m,n = mat.shape
    temp = np.ones((m,n+1))[0,:]
    vec = mat.A[0,:]
    temp[1:]=vec
    return temp    # Array

def minus_bias(mat):
    m,n = mat.shape
    temp = mat[0,1:]
    return temp



def logisticFunction(z):
    return 1.0/(1.0+np.exp(-z))

def logFuncDerivative(e):
    return np.exp(-e)/pow(np.exp(-e)+1,2)

class NeuralNet():
    def __init__(self,numInput,numHidden,numOutput,maxiterations,alpha=0.001):
        self.numInput = numInput
        self.numHidden = numHidden
        self.numOutput = numOutput
        self.maxiterations = maxiterations
        self.learn_rate = alpha
        self.act_input = np.mat(np.ones((1,self.numInput+1)))
        self.act_hidden =np.mat(np.ones((1,self.numHidden)))
        self.act_Output = 1.0
        # every loop train a pair of one query
        #previous activation levels of all neurons
        self.prev_actInput = 0
        self.prev_actHidden = 0
        self.prev_actOutput = 0
        # previous delta in the output and hidden
        self.prev_DeltaOutput = 0
        self.prev_DeltaHidden = 0
        # current delta in the output and hidden layer
        self.deltaOutput = 0
        self.deltaHidden = 0
        # init weights
        self.weight_hidden = np.mat(np.random.rand(self.numHidden,self.numInput+1))
        self.weight_output = np.mat(np.random.rand(self.numOutput,self.numHidden+1))
        self.prev_actHidden_plus = 0
        self.act_hidden_plus = 0
        self.prev_DeltaHidden_minus = 0
        self.deltaHidden_minus = 0
        self.errorRate =[]
    def propagate(self,inputs):
        self.prev_actInput = self.act_input
        # matrix
        self.act_input = inputs
        self.prev_actHidden = self.act_hidden
        # Array
        self.prev_actHidden_plus = add_bias(self.prev_actHidden)
        # 矩阵
        self.act_hidden = self.act_input*self.weight_hidden.T
        self.act_hidden = logisticFunction(self.act_hidden)

        self.prev_actOutput = self.act_Output
        # add bias >Array
        self.act_hidden_plus = add_bias(self.act_hidden)
        # act_Output matrix
        self.act_Output = logisticFunction(self.act_hidden_plus*self.weight_output.T)
        # float
        self.act_Output = float(self.act_Output)

    def computeOutputDelta(self):
        prop = 1.0/(1.0+np.exp(-(self.prev_actOutput -self.act_Output)))
        self.prev_DeltaOutput = logFuncDerivative(self.prev_actOutput)*(1.0-prop)
        self.deltaOutput = logFuncDerivative(self.act_Output)*(1.0-prop)

    def computeHiddenDelta(self):
        # 得到的是带偏置的hidden layer Delta
        self.prev_DeltaHidden =np.multiply(logFuncDerivative(self.prev_actHidden_plus),
                                           self.weight_output*(self.prev_DeltaOutput-self.deltaOutput))
        # 去掉偏置的hidden layer Delta
        self.prev_DeltaHidden_minus = minus_bias(self.prev_DeltaHidden)
        # + bias
        self.deltaHidden = np.multiply(logFuncDerivative(self.act_hidden_plus),
                                       self.weight_output*(self.prev_DeltaOutput-self.deltaOutput))
        # -bias  matrix
        self.deltaHidden_minus = minus_bias(self.deltaHidden)
    def updateWeights(self):
        self.weight_output = self.weight_output + self.learn_rate*(
            self.prev_DeltaOutput*self.prev_actHidden_plus- self.deltaOutput*self.act_hidden_plus)
        self.weight_hidden = self.weight_hidden + self.learn_rate*(
            self.prev_DeltaHidden_minus.T*self.prev_actInput - self.deltaHidden_minus.T*self.act_input)

    def backpropagate(self):
        self.computeOutputDelta()
        self.computeHiddenDelta()
        self.updateWeights()

    def trainModel(self,X,pairs):
        start = time.time()
        print ('Traning the neural network...')
        for epoch in range(self.maxiterations):
            print '### Epoch %d ###'%(epoch+1)
            for pair in pairs:
                self.propagate(X[pair[0]])
                self.propagate(X[pair[1]])
                self.backpropagate()
            missRate = self.countMissOfpairs(X,pairs)
            self.errorRate.append(missRate)

            print ('Error rate:%.10f'%missRate)
        m,s = divmod(time.time()-start,60)
        print 'Training took %d m %.1f s'%(m,s)

    def countMissOfpairs(self,X,pairs):
        miss = 0
        for pair in pairs:
            self.propagate(X[pair[0]])
            self.propagate(X[pair[1]])
            if self.prev_actOutput <= self.act_Output:
                miss = miss+1
        return miss /float(len(pairs))









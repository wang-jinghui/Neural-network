import  numpy as np
import time

class NNwork(object):
    def __init__(self,sizes):
        self.sizes = sizes
        self.numa_layers = len(sizes)
        self.weights = [np.random.randn(y,x+1) for x,y in zip(sizes[:-1],sizes[1:])]

    def feedforward(self, train_x):
        a = np.mat(train_x)
        for w in self.weights:
            z = a * w.T
            a = self.sigmoid(z)
            a = self.add_bias(a)
        y = self.reduce_bias(a)
        return y

    def BGD(self,train_x, train_y, epochs, alpha, lamda=0):
        start = time.time()


        numlist = []
        for i in xrange(epochs):
            self.update_weights(train_x,train_y,alpha,lamda)
            er = self.error_rate(train_x,train_y)
            numlist.append(er)
            print '%d epochs  :%d / 9219 '%(i+1,er)
            #cost = self.costFunction(train_x, train_y,lamda)
            #print '%d epochs over'%(i+1)
            #costlist.append(cost)
        return numlist

        m, s = divmod(time.time()-start, 60)
        print 'Training took %dm %.1fs'%(m, s)




    def update_weights(self,train_x,train_y,alpha,lamda):
        delta_w = self.backprop(train_x,train_y,lamda)
        self.weights = [w - alpha*nw for w,nw in zip(self.weights,delta_w)]

    def backprop(self,train_x,train_y,lamda):
        list_a = []
        list_ab = []
        list_e = []
        delta_w = [np.zeros(w.shape) for w in self.weights]
        m = len(train_y)
        y = np.mat(train_y)
        a = np.mat(train_x)
        list_ab.append(a)
        for w in self.weights:
            z = a * w.T
            a = self.sigmoid(z)
            list_a.append(a)
            a = self.add_bias(a)
            list_ab.append(a)
        e = (list_a[-1] - y)
        list_e.append(e)
        for i in xrange(2,self.numa_layers):
            e = e * self.weights[-i+1]
            e = self.reduce_bias(e)
            e = np.multiply(e,self.delta_sigmoid(list_a[-i]))
            list_e.append(e)
        list_e = self.revers(list_e)
        for l in xrange(len(self.weights)):
            for j in xrange(m):
                delta_w[l] = delta_w[l] + list_e[l][j].T*list_ab[l][j]
            delta_w[l] = delta_w[l]/m
        return delta_w

    def sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def delta_sigmoid(self,a):
        return np.multiply(a,(1-a))

    def add_bias(self,mat):
        m,n = mat.shape     #matrix
        temp = np.ones((m,n+1))  # array
        temp[:,1:]=mat         # array
        return np.mat(temp)

    def reduce_bias(self,mat):
        return mat[:,1:]

    def revers(self,l):
        temp = []
        for i in xrange(1,len(l)+1):
            temp.append(l[-i])
        return temp

    def error_rate(self,train_x, train_y):
        m = len(train_y)
        s = 0
        predict = self.predict(train_x)
        for i in xrange(m):
            if predict[i] == train_y[i].index(1.0):
                s = s+1

        return s

    def predict(self,train_x):
        pred = self.feedforward(train_x)
        predicted = []
        for p in pred:
            y = p.argmax()
            predicted.append(y)
        return predicted

    def costFunction(self, train_x, train_y, lamda):
        m = len(train_y)
        train_y = np.mat(train_y)
        a = self.feedforward(train_x)
        cost = np.sum(np.multiply(train_y, np.log(a))+np.multiply((1-train_y), np.log(1-a)))
        regular = []
        for w in self.weights:
            w = self.reduce_bias(w)
            w = np.sum(np.multiply(w,w))
            regular.append(w)

        cost = -1*cost/m + (lamda/(2*m))*sum(regular)
        return cost

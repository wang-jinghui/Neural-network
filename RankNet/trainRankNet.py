# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from process_data import *
from BPnetwork import *

train_x,train_y,train_query = loadDataSet('train.txt')
#val_x,val_y,val_query = loadDataSet('dataSet/vali.txt')
#test_x,test_y,test_query = loadDataSet('dataSet/test.txt')
train_pairs = extractPairs(train_y,train_query)
#val_pairs = extractPairs(val_y,val_query)
#test_pairs = extractPairs(test_y,test_query)
RankNet = NeuralNet(46,20,1,50,0.001)
RankNet.trainModel(train_x,train_pairs)
train_errorRate = RankNet.errorRate
#val_errorRate = RankNet.countMissOfpairs(val_x,val_pairs)
#test_errorRate = RankNet.countMissOfpairs(test_x,test_pairs)
plt.plot(train_errorRate)
plt.show()


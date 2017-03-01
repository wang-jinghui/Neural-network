# -*- coding:utf-8 -*-
'''learning to rank using gradient descent'''
import numpy as np
def loadDataSet(path):
    '''LETOR dataset'''
    X = []
    Y = []
    Query = [] #<query_id><document_id><inc><prob>
    print 'loading train dataset'
    with open(path,'r') as f:
        for line in f:
            line = line.split()
            Y.append(int(line[0]))
            X.append(extractFeatures(line))
            Query.append(extractQueryData(line))
    print 'load %d examples from dataset...'%(len(X))
    return (np.mat(X),Y,Query)

def extractFeatures(line):
    features = [1.0]
    for i in xrange(2,48):
        features.append(float(line[i].split(':')[1]))
    return features

def extractQueryData(line):
    queryFeatures = [line[1].split(':')[1]]
    queryFeatures.append(line[50])
    queryFeatures.append(line[53])
    queryFeatures.append(line[56])
    return queryFeatures

# 同一个查询的文档对,每个文档对相关度高的在前
def extractPairs(Y,Query):
    pairs=[]
    for i in xrange(len(Query)):
        for j in xrange(i+1,len(Query)):
            if Query[i][0] != Query[j][0]:
                break
            if Query[i][0] == Query[j][0] and Y[i] != Y[j]:
                if Y[i] > Y[j]:
                    pairs.append([i,j])
                else:
                    pairs.append([j,i])
    print 'found %d document pairs'%(len(pairs))
    return pairs















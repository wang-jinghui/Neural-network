# -*- coding:utf-8 -*-
# process data
# 数据集来自微软亚洲研究院公开数据集LETOR 4.0
#

import numpy as np


def processData(filename):
    X = []
    Y = []
    Query = []
    with open(filename, 'r') as f:
        for row in f:
            line = row.split()
            Y.append(int(line[0]))
            X.append(extractFeatures(line))
            Query.append(extractQuery(line))
    return (X, Y, Query)

def extractFeatures(line):
    features = []
    for i in range(2, 48):
        features.append(float(line[i].split(':')[1]))
    return features

def extractQuery(line):
    return int(line[1].split(':')[1])


def predict_socre(predict, y):
    num = 0
    for p, y in zip(predict, y):
        if p == y:
            num +=1
    return  num

 

 
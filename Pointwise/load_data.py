# process data
import numpy as np
import math

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

def costFunction(p, y):
    p = np.array(p)
    y = np.array(y)
    m = len(p)
    e = p-y
    cost = np.sum(np.multiply(e,e))/float(m)
    return cost


def GetRankedLabels(Yrankscores, Y, QueryIds):
    dict = {}
    for i in range(len(QueryIds)):
        if QueryIds[i] in dict:
            dict[QueryIds[i]].append((Yrankscores[i], Y[i]))
        else:
            vec = []
            vec.append((Yrankscores[i], Y[i]))
            dict[QueryIds[i]] = vec

    retdict = {}
    for queryid in dict.keys():
        retdict[queryid] = [tuple[1] for tuple in sorted(dict[queryid], reverse=True)]
    return retdict


def GetPrecisionatK(eval, k):
    precisionatk = 0.0
    for queryid in eval:
        precisionatk += float(np.sum(np.array(eval[queryid][:k], dtype=int) ==2)) / float(k)
    precisionatk /= float(len(eval))
    return precisionatk


def GetMAP(eval):
    map = 0.0
    for queryid in eval:
        denominator = np.sum(np.array(eval[queryid], dtype=int) ==2)
        if denominator == 0: denominator = 1
        val = 0.0
        for pos in range(len(eval[queryid])):
            if eval[queryid][pos] ==2 :
                val += float(np.sum(np.array(eval[queryid][:pos + 1], dtype=int) ==2)) / float(pos + 1)
        val /= denominator
        map += val
    map /= float(len(eval))
    return map


def GetNDCGatK(eval, k):
    ndcgatk = 0.0
    discounts = np.zeros(k)
    for i in range(k): discounts[i] = float(1.0) / math.log(i + 2, 2)

    for queryid in eval:
        gain = np.array([2 ** val - 1 for val in eval[queryid][:k]])
        idealgain = np.array([2 ** val - 1 for val in sorted(eval[queryid][:k], reverse=True)])
        denominator = np.dot(idealgain,discounts)
        if denominator == 0.0: denominator = 1.0
        ndcgatk += float(np.dot(gain, discounts)) / float(denominator)
    ndcgatk /= float(len(eval))
    return ndcgatk

def GaussianProce(a):
    return a**(1.0/2)
import pandas as pd
import numpy as np
import scipy.io as sio
import heapq

def calculatePrecisionAtK(res, data, K):
    # print(res)
    # print(data)
    shapes = res.shape
    samples = shapes[0]
    l = []

    data = data[:,0]
    for i in range(samples):
        if data[i]==1:
            l.append(i)
    anomasize = len(l)

    # print(shapes)
    anslist = []
    if len(shapes)==2:
        for i in range(samples):
            anslist.append([res[i][0], i])
    else:
        for i in range(samples):
            anslist.append([res[i], i])
    out_data = sorted(anslist, key = lambda x:x[0])

    count = 0
    for i in range(K):
        item = out_data[i]
        if(item[1] in l):
            count+=1
    return count/anomasize
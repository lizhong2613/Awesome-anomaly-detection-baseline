import pandas as pd
import numpy as np
import scipy.io as sio
import heapq
import csv

dataset_path = "dataset/IMDB.mat"
data = sio.loadmat(dataset_path)
data = data['gnd']

res = np.load("imdbresult.npy")

print(res.sum())
# print(data)

l = []

data = data[:,0]
for i in range(4780):
    if data[i]==1:
        l.append(i)

print(len(l))
anslist = []
for i in range(4780):
    anslist.append([res[i][0], i])
# print(anslist)
out_data = sorted(anslist, key = lambda x:x[0], reverse=True)
# print("\n", out_data, "\n")

count = 0
for i in range(300):
    item = out_data[i]
    if(item[1] in l):
        count+=1
print(count)
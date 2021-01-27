import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import normA
import DominantModel as Dominant
from getData import GetData

def main(iterations = 300):

    # get data
    datagetter = GetData("data/Amazon.mat")
    A, X, gnd = datagetter.readFile()
    Anorm = normA.noramlization(A)
    A = torch.tensor(A, dtype=torch.float32)
    X = torch.tensor(X, dtype=torch.float32)
    gnd = torch.tensor(gnd, dtype=torch.float32)
    Anorm = torch.tensor(Anorm, dtype=torch.float32)
    samples = datagetter.returnSamples()
    attributes = datagetter.returnAttributes()

    # model
    dominant = Dominant.DominantModel(Anorm, attributes)
    gd = torch.optim.Adam(dominant.parameters())

    for iter in range(iterations):
        reconstructionStructure, reconstructionAttribute = dominant(X)

if __name__=="__main__":
    main()
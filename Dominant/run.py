import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import normA
import DominantModel as Dominant
from getData import GetData

def main(alpha = 0.5, iterations = 300):

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


    # cuda
    if torch.cuda.is_available():
        A = A.cuda()
        X = X.cuda()
        gnd = gnd.cuda()
        Anorm = Anorm.cuda()
        dominant = Dominant.DominantModel(Anorm, attributes)
        dominant = dominant.cuda()
    else:
        dominant = Dominant.DominantModel(Anorm, attributes)

    gd = torch.optim.Adam(dominant.parameters(), lr=0.005) 
    # print(dominant)

    # training
    for iter in range(iterations):
        reconstructionStructure, reconstructionAttribute = dominant(X)
        loss = alpha * torch.norm(reconstructionStructure - A) + (1 - alpha) * torch.norm(reconstructionAttribute - X)

        gd.zero_grad()
        loss.backward()
        gd.step()

    # get score
    if torch.cuda.is_available():
        structureError = (reconstructionStructure - A).cpu().detach().numpy()
        attributeError = (reconstructionAttribute - X).cpu().detach().numpy()
    else:
        structureError = (reconstructionStructure - A).detach().numpy()
        attributeError = (reconstructionAttribute - X).detach().numpy()
    structureLoss = np.linalg.norm(structureError, axis=1, keepdims=True)
    attributeLoss = np.linalg.norm(attributeError, axis=1, keepdims=True)
    score = alpha * structureLoss + (1 - alpha) * attributeLoss
    target = gnd.reshape((-1))
    target = target.tolist()
    score = score.tolist()
    print(roc_auc_score(target, score))

if __name__=="__main__":
    main()
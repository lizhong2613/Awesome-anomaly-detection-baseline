import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import GraphConv

class DominantModel(nn.Module):
    def __init__(self, A_norm, dim_in, dim_out=16):
        super(DominantModel, self).__init__()
        self.gcn1 = GraphConv.GCN(A_norm, dim_in, 64)
        self.gcn2 = GraphConv.GCN(A_norm, 64, 32)
        self.gcn3 = GraphConv.GCN(A_norm, 32, dim_out)
        self.deconv = GraphConv.GCN(A_norm, dim_out, dim_in)

    def forward(self, X):
        X = self.gcn1(X)
        X = self.gcn2(X)
        Z = self.gcn3(X)
        return Z.mm(Z.transpose(0, 1)), self.deconv(Z)
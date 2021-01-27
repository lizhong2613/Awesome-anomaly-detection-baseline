import torch
import numpy as np
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    '''
    Z = AXW
    '''

    def __init__(self, A_norm, dim_in, dim_out):
        super(GCN, self).__init__()
        self.A_norm = A_norm
        self.fc = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, X):
        '''
        1 layer GCN
        '''
        return F.relu(self.fc(self.A_norm.mm(X)))

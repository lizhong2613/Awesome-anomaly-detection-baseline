import numpy as np

def noramlization(A):
    A_norm = A + np.identity(A.shape[0])
    vec_d = np.sum(A_norm,axis=1)
    D = np.diag(vec_d**(-0.5))
    print(D)
    return np.matmul(np.matmul(D, A_norm), D)

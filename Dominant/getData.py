import numpy as np
import scipy.io as sio

class getData(object):
    '''
    return data of A, X and gnd
    '''
    def __init__(self, path):
        self.path = path
        self.shapeA = None
        self.shapeX = None
        self.shapegnd = None
        self.samples = None
        self.attributes = None

    def __str__(self):
        return "Shape of A:\t" + str(self.shapeA) + "\nShape of X: \t" + str(self.shapeX) + "\nShape of gnd: \t" + str(self.shapegnd)

    def readFile(self):
        data = sio.loadmat(self.path)
        self.A = data['A']
        self.A = np.max(self.A, self.A.T)
        self.X = data['X']
        self.gnd = data['gnd']
        checkshape = self.checkShape()
        if checkshape != "match!":
            print(checkshape)
            return
        return self.A, self.X, self.gnd

    def checkShape(self):
        self.shapeA = self.A.shape
        self.shapeX = self.X.shape
        self.shapegnd = self.gnd.shape
        if self.shapeA[0] != self.shapeA[1]:
            return "shape of A: " + str(self.shapeA[0]) + " and " + str(self.shapeA[1]) + " not matched!"
        if self.shapeA[0] != self.shapeX[0]:
            return "shape of A and X: " + str(self.shapeA[0]) + " and " + str(self.shapeX[0]) + " not matched!"
        if self.shapeA[0] != self.shapegnd[0]:
            return "shape of A and gnd: " + str(self.shapeA[0]) + " and " + str(self.shapegnd[0]) + " not matched!"
        else:
            return "match!"

# a = getData("data/Amazon.mat")
# a.readFile()
# print(a)
# A, X, gnd = a.readFile()
# print(A)
# print(X)
# print(gnd)
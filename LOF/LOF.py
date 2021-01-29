import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import calculateAUC
import getData

data_generator = getData.GetData('data/Amazon.mat')
A, X, gnd = data_generator.readFile()

lof = LocalOutlierFactor(n_neighbors=2)
lof.fit_predict(X)

anom_score = -(lof.negative_outlier_factor_)

print(calculateAUC.getAUC(score=anom_score, gnd=gnd))
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import getData

data_generator = getData.GetData('data/Disney.mat')
A, X, gnd = data_generator.readFile()

lof = LocalOutlierFactor(n_neighbors=2)
lof.fit_predict(X)

anom_score = -(lof.negative_outlier_factor_)
anom_score = anom_score.reshape((-1, 1))
target = gnd.reshape((-1))
target = target.tolist()
anom_score = anom_score.tolist()
print(roc_auc_score(target, anom_score))
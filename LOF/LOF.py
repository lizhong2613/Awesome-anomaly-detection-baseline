import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import calculateAUC
import getData
from recallAtK import calculateRecallAtK

def main(K:int):
    data_generator = getData.GetData('data/Amazon.mat')
    A, X, gnd = data_generator.readFile()

    lof = LocalOutlierFactor(n_neighbors=2)
    lof.fit_predict(X)

    anom_score = -(lof.negative_outlier_factor_)

    RecallatK = calculateRecallAtK(anom_score, gnd, K)
    print("Recall @ {}: \t{}".format(K, RecallatK))

    print("AUC value: \t{}".format(calculateAUC.getAUC(score=anom_score, gnd=gnd)))

if __name__ == "__main__":
    main(300)
from sklearn.metrics import roc_auc_score

def getAUC(score, gnd):
    """
    shape of score: (n x 1)
    shape of gnd: (n x 1)
    """
    score = score.reshape((-1, 1))
    target = gnd.reshape((-1))
    target = target.tolist()
    score = score.tolist()
    return roc_auc_score(target, score)
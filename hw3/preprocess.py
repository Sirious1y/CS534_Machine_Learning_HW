import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


# 1c
def compute_correlation(x, corrtype):
    df = pd.DataFrame(x)
    corr = df.corr(method=corrtype)
    return corr.to_numpy()


# 1d
def rank_correlation(x, y):
    x = np.transpose(x)
    corrs = []
    for column in x:
        covar = np.cov(column.astype(float), y)
        # print(covar)
        corr = covar[0, 1]/np.sqrt(covar[0, 0] * covar[1, 1])
        corrs.append(corr)
    corrs = np.abs(corrs)
    # print(corrs)
    sorted_idx = np.argsort(corrs)[::-1]
    return sorted_idx


# 1e
def rank_mutual(x, y):
    mutual_info = np.abs(mutual_info_classif(x, y))
    # print(mutual_info)
    sorted_idx = np.argsort(mutual_info)[::-1]
    return sorted_idx

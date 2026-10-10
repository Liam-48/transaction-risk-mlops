import numpy as np

def negative_to_nan(X):
    return np.where(X < 0, np.nan, X)
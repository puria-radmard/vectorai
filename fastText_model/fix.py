import pandas as pd
import numpy as np

X = pd.DataFrame([1, 2, 3, None, 4, 5, None, 6])
y = pd.Series([1, 1, 1, 0, 1, 1, 0, 1])


def remove_nones(X, y):
    inds = X[X[0].notnull()].index
    print(inds)
    X_out = X.iloc[inds]
    print(X_out)
    y_out = y[inds]
    print(y_out)
    #return X_out, y_out

print(remove_nones(X, y))

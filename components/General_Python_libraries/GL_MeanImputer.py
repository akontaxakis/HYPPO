import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class GL__SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):  # Default tol value is set to 0.001
        self.strategy = strategy

    def fit(self, X, y=None, ):
        self.mean_ = np.nanmean(X, axis=0)
        return self

    def transform(self, X):

        X = np.array(X).astype('float32')
        imputed = np.where(np.isnan(X), self.mean_, X)
        return np.array(imputed)

    def fit_transform(self, X, y=None):
        X = np.array(X).astype('float32')
        y = np.array(y).astype('float32')
        self.fit(X, y)
        return self.transform(X)
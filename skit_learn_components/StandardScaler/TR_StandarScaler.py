import torch
from sklearn.base import BaseEstimator, TransformerMixin

class TR__StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X, y=None):
        X = torch.tensor(X).float()
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)
        return self

    def transform(self, X):
        X = torch.tensor(X).float()
        X_scaled = (X - self.mean) / self.std
        return X_scaled.numpy()
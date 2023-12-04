
import torch
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TR__SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X, y=None):
        X = torch.tensor(X).float()
        if self.strategy == 'mean':
            self.statistics_ = torch.mean(X, dim=0)
        elif self.strategy == 'median':
            self.statistics_, _ = torch.median(X, dim=0)
        else:
            raise ValueError("Can only use these strategies: 'mean', 'median'")
        return self

    def transform(self, X):
        X = torch.tensor(X).float()
        if torch.isnan(X).any():
            X[torch.isnan(X)] = self.statistics_[torch.isnan(X)]
        return X.numpy()
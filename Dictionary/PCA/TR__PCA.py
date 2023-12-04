import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

class TR__PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, X, y=None):
        X = torch.tensor(X).float()
        self.mean = torch.mean(X, dim=0)
        X = X - self.mean

        cov_matrix = torch.mm(X.t(), X) / X.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)  # Use eigh() here
        eigenvectors = eigenvectors[:, :self.n_components]
        self.eigenvectors = eigenvectors

        return self

    def transform(self, X):
        X = torch.tensor(X).float()
        X = X - self.mean
        X_pca = torch.mm(X, self.eigenvectors)
        X_pca=X_pca.numpy()
        X_pca = np.nan_to_num(X_pca)
        return X_pca
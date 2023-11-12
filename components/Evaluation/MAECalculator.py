from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error

class MAECalculator(BaseEstimator, TransformerMixin):
    def __init__(self, y_true=None):
        self.y_true = y_true

    def fit(self, X, y=None):
        self.y_true = X
        return self

    def score(self, X):
        mae = mean_absolute_error(self.y_true, X)
        return mae
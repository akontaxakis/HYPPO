from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error

class MSECalculator(BaseEstimator, TransformerMixin):
    def __init__(self, y_true=None):
        self.y_true = y_true

    def fit(self, X, y=None):
        self.y_true = X
        return self

    def score(self, X):
        mse = mean_squared_error(self.y_true, X)
        return mse
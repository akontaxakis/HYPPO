from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score, accuracy_score


class AccuracyCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, y_true):
        self.y_true = y_true

    def fit(self, X, y=None):
        self.y_true = X
        return self

    def score(self, X):
        accuracy = accuracy_score(self.y_true, X)
        return accuracy



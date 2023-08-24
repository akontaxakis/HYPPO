from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import f1_score

class F1ScoreCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, y_true):
        self.y_true = y_true

    def fit(self, X, y=None):
        self.y_true = X
        return self

    def score(self, X):
        f1 = f1_score(self.y_true, X)
        print("F1 Score: ", f1)
        return X
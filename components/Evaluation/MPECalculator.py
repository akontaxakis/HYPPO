import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class MPECalculator(BaseEstimator, TransformerMixin):
    def __init__(self, y_true=None):
        self.y_true = y_true

    def fit(self, X, y=None):
        # X is expected to be the true values here
        self.y_true = check_array(X, ensure_2d=False)
        # Ensure that we have no zeros in the true values array to avoid division by zero
        if (self.y_true == 0).any():
            raise ValueError("True values contain zero(s), which would result in division by zero when calculating MAPE.")
        return self

    def score(self, X):
        # X is expected to be the predictions here
        check_is_fitted(self, 'y_true')
        X = check_array(X, ensure_2d=False)
        mape = np.mean(np.abs((self.y_true - X) / self.y_true)) * 100
        return mape
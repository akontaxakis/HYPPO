import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

class GL__PolynomialFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X, y=None):
        # Validate the input array
        X = check_array(X, accept_sparse=True, dtype=[np.float64, np.float32], order='F', copy=True)
        self.n_input_features_ = X.shape[1]
        return self

    def transform(self, X):
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X, accept_sparse=True, dtype=[np.float64, np.float32], order='F', copy=True)

        # Initialize the output array
        n_samples, n_features = X.shape
        if self.include_bias:
            out = np.ones((n_samples, 1), dtype=X.dtype, order='F')
        else:
            out = np.empty((n_samples, 0), dtype=X.dtype, order='F')

        # Compute the polynomial features
        for degree in range(1, self.degree + 1):
            out = np.hstack((out, X ** degree))

        return out

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

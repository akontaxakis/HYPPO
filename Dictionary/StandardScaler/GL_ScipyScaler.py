import numpy as np
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin


from scipy.stats import zscore
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class GL__StScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Calculate the standard deviation per feature
        X  = X.astype(float)

        self.feature_std = np.std(X, axis=0)
        return self

    def transform(self, X):
        # If any feature has a standard deviation of zero, replace the entire feature by zero
        # Otherwise, standardize it
        standardized_X = np.where(self.feature_std == 0, 0, (X - np.mean(X, axis=0)) / self.feature_std)
        return standardized_X

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
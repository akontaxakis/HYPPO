from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import pairwise_distances
from scipy.stats import mode
import numpy as np

import numpy as np
from scipy.spatial import distance
from sklearn.base import BaseEstimator, RegressorMixin

class GL__KNeighborsRegressor(BaseEstimator, RegressorMixin):
        def __init__(self, n_neighbors=5, weights='uniform'):
            self.n_neighbors = n_neighbors
            self.weights = weights
            self.X_train = None
            self.y_train = None

        def fit(self, X, y):
            X = X.astype('float')
            y = y.astype('float')

            self.X_train = X
            self.y_train = y
            return self

        def predict(self, X):
            X = X.astype('float')
            # Check if X_train contains only numeric data.

            if not np.issubdtype(self.X_train.dtype, np.number):
                raise ValueError("X_train contains non-numeric data.")

            # Check if X contains only numeric data.
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError("X contains non-numeric data.")

            predictions = []
            X = X.astype('float')
            for x in X:
                # Compute the distance from x to all points in X_train
                dists = distance.cdist(self.X_train, [x]).flatten()
                # Get the indices of the n_neighbors nearest neighbors
                if self.n_neighbors > len(dists):
                    # In case n_neighbors is more than training samples
                    neighbors_indices = np.argsort(dists)
                else:
                    neighbors_indices = np.argsort(dists)[:self.n_neighbors]
                # Calculate the weighted mean of the nearest neighbors
                if self.weights == 'distance':
                    # Use the inverse of the distance as weights
                    weights = 1 / (dists[neighbors_indices] + 1e-5)  # add a small constant to avoid division by zero
                    weighted_average = np.dot(weights, self.y_train[neighbors_indices]) / np.sum(weights)
                    predictions.append(weighted_average)
                else:
                    # Simple average (uniform weights)
                    predictions.append(np.mean(self.y_train[neighbors_indices]))
            return np.array(predictions).flatten()


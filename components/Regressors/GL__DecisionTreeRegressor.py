import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import concurrent.futures


class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class GL__DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, min_samples_split=25, max_depth=None, min_samples_leaf=25):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        return self

    def _build_tree(self, X, y, current_depth=0):
        num_samples, num_features = X.shape
        if num_samples < self.min_samples_split or (self.max_depth is not None and current_depth >= self.max_depth):
            return DecisionNode(value=np.mean(y))

        best_split = self._get_best_split(X, y, num_samples, num_features)
        if best_split["value"] is not None:
            return DecisionNode(value=best_split["value"])

        left_subtree = self._build_tree(X[best_split["dataset_left"]], y[best_split["dataset_left"]], current_depth + 1)
        right_subtree = self._build_tree(X[best_split["dataset_right"]], y[best_split["dataset_right"]],
                                         current_depth + 1)
        return DecisionNode(feature_index=best_split["feature_index"], threshold=best_split["threshold"],
                            left=left_subtree, right=right_subtree)

    def _get_best_split(self, X, y, num_samples, num_features):
        best_split = {}
        min_mse = np.inf

        # Only consider a random subset of the features
        features = np.random.choice(num_features, int(np.sqrt(num_features)), replace=False)

        # Using ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._evaluate_split, X, y, feature_index, threshold): feature_index
                       for feature_index in features
                       for threshold in np.unique(X[:, feature_index])}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result["mse"] < min_mse:
                    min_mse = result["mse"]
                    best_split = result["best_split"]

        if not best_split:
            return {"value": np.mean(y)}

        return best_split

    def _evaluate_split(self, X, y, feature_index, threshold):
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]

        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return {"mse": np.inf, "best_split": None}

        y_left, y_right = y[left_indices], y[right_indices]
        mse = self._calculate_mse(y_left, np.mean(y_left)) + self._calculate_mse(y_right, np.mean(y_right))

        return {
            "mse": mse,
            "best_split": {
                "feature_index": feature_index,
                "threshold": threshold,
                "dataset_left": left_indices,
                "dataset_right": right_indices,
                "value": None
            }
        }

    def _calculate_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def predict(self, X):
        return np.array([self._predict_value(x, self.root) for x in X])

    def _predict_value(self, x, node):
        if node.value is not None:  # It's a leaf node
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_value(x, node.left)
        else:
            return self._predict_value(x, node.right)

# Example usage:
# from sklearn.datasets import make_regression
# X, y = make_regression(n_samples=100, n_features=4, noise=0.1)
# reg = GL__DecisionTreeRegressor(min_samples_split=5, max_depth=10, min_samples_leaf=4)
# reg.fit(X, y)
# predictions = reg.predict(X)

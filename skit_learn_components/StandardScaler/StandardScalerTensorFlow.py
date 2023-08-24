import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class TF__StandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_ = None
        self.var_ = None
        self.n_samples_seen_ = None

    def fit(self, X, y=None):
        self.mean_ = tf.reduce_mean(X, axis=None)
        self.var_ = tf.math.reduce_variance(X, axis=None)
        return self

    def transform(self, X):
        X = tf.convert_to_tensor(X)
        return ((X - self.mean_) / tf.sqrt(self.var_)).numpy()

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
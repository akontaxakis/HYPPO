import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

class TF__SimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean'):
        self.strategy = strategy

    def fit(self, X, y=None):
        if self.strategy == 'mean':
            self.fill = tf.reduce_mean(tf.boolean_mask(X, tf.math.is_finite(X)))
        elif self.strategy == 'median':
            self.fill = tf.stats.percentile(tf.boolean_mask(X, tf.math.is_finite(X)), 50.0)
        elif self.strategy == 'constant':
            self.fill = 0.0
        else:
            raise ValueError('Invalid strategy')
        return self

    def transform(self, X):
        return tf.where(tf.math.is_finite(X), X, self.fill).numpy()

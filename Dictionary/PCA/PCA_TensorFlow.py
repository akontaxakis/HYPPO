import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

class TF__PCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.mean = tf.reduce_mean(X, axis=0)
        X = X - self.mean
        X = tf.cast(X, tf.float32)
        # Calculate covariance matrix
        cov = tf.linalg.matmul(X, X, transpose_a=True) / tf.cast(tf.shape(X)[0], tf.float32)

        # Calculate eigenvalues and eigenvectors for the covariance matrix
        self.eigenvalues, self.eigenvectors = tf.linalg.eigh(cov)

        # Sort eigenvectors by eigenvalues in descending order
        indices = tf.argsort(self.eigenvalues, direction='DESCENDING')
        self.eigenvectors = tf.gather(self.eigenvectors, indices)

        return self

    def transform(self, X):
        X = X - self.mean
        X = tf.cast(X, tf.float32)
        return tf.linalg.matmul(X, self.eigenvectors[:, :self.n_components]).numpy()

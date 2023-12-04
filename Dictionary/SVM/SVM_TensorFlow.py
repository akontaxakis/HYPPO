import tensorflow as tf
from tensorflow import keras
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class TF__LnearSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs=100, lr=0.01):
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None

    def hinge_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        return tf.reduce_mean(tf.maximum(1. - y_true * y_pred, 0.))

    def fit(self, X, y):

        if self.model is None:
            self.model = keras.Sequential()
            self.model.add(keras.layers.Dense(1, activation='linear', input_shape=(X.shape[1],)))
            self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=self.lr), loss=self.hinge_loss)

        self.model.fit(X, y, epochs=self.n_epochs, verbose=0)
        return self

    def predict(self, X):
        return np.sign(self.model.predict(X)).flatten()

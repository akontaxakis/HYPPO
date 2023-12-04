import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class TF__LinearRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, epochs=10, batch_size=10, learning_rate=0.01):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_ = None
        self.history_ = None

    def fit(self, X, y):
        # Ensure that X and y are numpy arrays
        X, y = np.array(X), np.array(y)

        X = np.array(X).astype('float32')
        y = np.array(y).astype('float32')

        # Define the TensorFlow model
        self.model_ = Sequential()
        self.model_.add(Dense(units=1, input_shape=(X.shape[1],), activation='linear'))

        # Compile the model
        self.model_.compile(optimizer=Adam(learning_rate=self.learning_rate),
                            loss='mean_squared_error')

        # Fit the model
        self.history_ = self.model_.fit(X, y, epochs=self.epochs,
                                        batch_size=self.batch_size, verbose=0)

        return self

    def predict(self, X):
        X = np.array(X).astype('float32')

        # Check if the model is fitted
        if self.model_ is None:
            raise NotFittedError(
                "This TFLinearRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # Perform prediction
        return  self.model_.predict(np.array(X)).flatten()


    def score(self, X, y):
        # Check if the model is fitted
        if self.model_ is None:
            raise NotFittedError(
                "This TFLinearRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

        # Evaluate the model
        results = self.model_.evaluate(np.array(X), np.array(y), verbose=0)

        # The `evaluate` method returns the loss by default, which is mean squared error for regression models
        # scikit-learn's `score` method is expected to return the coefficient of determination R^2 of the prediction
        # which is not directly provided by TensorFlow's evaluate method, so we'll need to compute it.
        y_pred = self.model_.predict(np.array(X))
        ss_res = np.sum(np.square(np.array(y) - y_pred))
        ss_tot = np.sum(np.square(np.array(y) - np.mean(y)))
        r2_score = 1 - (ss_res / ss_tot)

        return r2_score

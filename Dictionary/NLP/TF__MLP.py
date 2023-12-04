from keras import Sequential
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.python.layers.core import Dense
from sklearn.metrics import accuracy_score


class TF__MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs=10):
        self.n_epochs = n_epochs
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def fit(self, X, y):
        self.model.compile(optimizer='sgd', loss='binary_crossentropy')
        self.model.fit(X, y, epochs=self.n_epochs)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return (y_pred > 0.5).astype('int32')

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
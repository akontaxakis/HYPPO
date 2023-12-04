import torch
from sklearn.metrics import accuracy_score
from torch import nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin

class TR__MLP(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs=10):
        self.n_epochs = n_epochs
        self.criterion = nn.BCELoss()

    def fit(self, X, y):
        self.model = nn.Sequential(
            nn.Linear(X.shape[1], 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

        y = y.reshape(-1, 1)  # reshape y if necessary
        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(torch.FloatTensor(X))
            loss = self.criterion(outputs, torch.FloatTensor(y))
            loss.backward()
            self.optimizer.step()

        return self

    def predict(self, X):
        with torch.no_grad():
            y_pred = self.model(torch.FloatTensor(X))
        return (y_pred.numpy() > 0.5).astype(int)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
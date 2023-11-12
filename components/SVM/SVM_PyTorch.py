import torch
from sklearn.base import BaseEstimator, ClassifierMixin

class TR__LinearSVC(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs=100, lr=0.1):
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None

    def hinge_loss(self, output, target):
        return torch.mean(torch.clamp(1 - output.t() * target, min=0))  # hinge loss

    def fit(self, X, y):
        X, y = torch.tensor(X).float(), torch.tensor(y).float()

        if self.model is None:
            self.model = torch.nn.Linear(X.size(1), 1)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        for _ in range(self.n_epochs):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.hinge_loss(output, y)
            loss.backward()
            self.optimizer.step()
        return self

    def predict(self, X):
        with torch.no_grad():
            X = torch.tensor(X).float()
            output = self.model(X)
        return output.numpy().flatten().round()



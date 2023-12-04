from sklearn.base import BaseEstimator, RegressorMixin
from pyglmnet import GLM


class GL__LassoGLM(BaseEstimator, RegressorMixin):
    def __init__(self, distr='gaussian', alpha=1.0, reg_lambda=0.1):
        self.distr = distr
        self.alpha = alpha
        self.reg_lambda = reg_lambda
        self.model_ = GLM(distr=self.distr, alpha=self.alpha, reg_lambda=self.reg_lambda)

    def fit(self, X, y):
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
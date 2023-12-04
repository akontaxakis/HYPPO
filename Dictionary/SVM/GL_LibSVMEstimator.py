import numpy as np
from libsvm.svmutil import svm_train, svm_predict
from sklearn.base import BaseEstimator, ClassifierMixin


class GL__SVMLibEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, tol=0.001, kernel='linear'):  # Default tol value is set to 0.001
        self.C = C
        self.tol = tol
        self.kernel = kernel
    def fit(self, X, y):
        options = '-t 0 -c {} -e {} -h 1'.format(self.C, self.tol)  # Including the tolerance in the options
        self.model_ = svm_train(y.tolist(), X.tolist(), options)
        return self

    def predict(self, X):
        p_label, _, _ = svm_predict([0] * len(X), X.tolist(), self.model_, '-q')
        return np.array(p_label)
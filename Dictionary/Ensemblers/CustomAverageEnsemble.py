import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.ensemble import VotingRegressor
import random
import os

# Custom estimator that selects, loads, and fits a voting ensemble
from sklearn.utils.validation import check_is_fitted


class GL__AverageRegressorLoader(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models_dir, n_models=6):
        self.models_dir = models_dir
        self.n_models = n_models
        self.regressors = None
        self.selected_model_files = []


    def fit(self, X, y):
        model_files = os.listdir(self.models_dir)
        self.selected_model_files = random.sample(model_files, self.n_models)
        # Load the models
        self.regressors = [(f'model_{i}', joblib.load(os.path.join(self.models_dir, path)))
                       for i, path in enumerate(self.selected_model_files)]


    def predict(self, X):
        predictions = np.column_stack([regr.predict(X) for _, regr in self.regressors])
        return np.mean(predictions, axis=1)


    def get_selected_models(self):
        # Check if the voting regressor is fitted

        # Return the selected models
        return self.selected_model_files


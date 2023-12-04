import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingRegressor
import random
import os

# Custom estimator that selects, loads, and fits a voting ensemble
from sklearn.utils.validation import check_is_fitted


class VotingRegressorLoader(BaseEstimator, ClassifierMixin):
    def __init__(self, models_dir, n_models=6):
        self.models_dir = models_dir
        self.n_models = n_models
        self.voting_regressor = None
        self.selected_model_files = []
        # Load the models

    def fit(self, X, y):
        model_files = os.listdir(self.models_dir)
        self.selected_model_files = random.sample(model_files, self.n_models)

        # Load the models
        self.models = [(f'model_{i}', joblib.load(os.path.join(self.models_dir, path)))
                       for i, path in enumerate(self.selected_model_files)]

        # Create the VotingRegressor with the selected models
        self.voting_regressor = VotingRegressor(estimators=self.models)

        # Fit the VotingRegressor
        self.voting_regressor.fit(X, y)

        return self

    def predict(self, X):
        # Check if the voting regressor is fitted
        check_is_fitted(self, 'voting_regressor')

        # Predict
        return self.voting_regressor.predict(X)

    def get_selected_models(self):
        # Check if the voting regressor is fitted

        # Return the selected models
        return self.selected_model_files


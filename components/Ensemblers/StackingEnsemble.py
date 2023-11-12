import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import Pipeline
import random
import os

# Custom transformer to select and load a random subset of models
from sklearn.utils.validation import check_is_fitted


class SetackingEnsemble(BaseEstimator, TransformerMixin):
    def __init__(self, models_dir, n_models=6, final_estimator=None):
        self.models_dir = models_dir
        self.n_models = n_models
        self.final_estimator = final_estimator
        self.selected_model_files = []

    def fit(self, X, y=None):
        model_files = os.listdir(self.models_dir)
        self.selected_model_files = random.sample(model_files, self.n_models)
        # Load the models
        self.base_models = [(f'model_{i}', joblib.load(os.path.join(self.models_dir, path)))
                            for i, path in enumerate(self.selected_model_files)]
        # Create and fit the StackingRegressor
        self.stacking_regressor = StackingRegressor(
            estimators=self.base_models, final_estimator=self.final_estimator
        )
        self.stacking_regressor.fit(X, y)
        return self

    def get_selected_models(self):
        # Check if the voting regressor is fitted

        # Return the selected models
        return self.selected_model_files


    def predict(self, X):
        # Proxy method to make the class compatible with the Pipeline
        return self.stacking_regressor.predict(X)
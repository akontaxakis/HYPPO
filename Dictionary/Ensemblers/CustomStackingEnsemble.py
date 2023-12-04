import os
import joblib
import random
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_squared_error


class GL__StackingRegressorLoader(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models_dir, n_models=6, validation_data=None):
        self.models_dir = models_dir
        self.n_models = n_models
        self.base_regressors = None
        self.selected_model_files = []
        self.validation_data = validation_data
        self.final_estimator_ = None

    def fit(self, X, y):
        # Check if directory exists and there are enough model files
        if not os.path.isdir(self.models_dir):
            raise FileNotFoundError(f"The specified directory {self.models_dir} does not exist.")

        model_files = os.listdir(self.models_dir)
        if len(model_files) < self.n_models:
            raise ValueError("Not enough models in the directory to load the requested number of models.")

        # Randomly select model files
        self.selected_model_files = random.sample(model_files, self.n_models)

        # Load the models and their validation predictions
        self.base_regressors = []
        for i, path in enumerate(self.selected_model_files):
            model = joblib.load(os.path.join(self.models_dir, path))
            self.base_regressors.append((f'model_{i}', model))

        # Select the best model based on the validation set
        if self.validation_data is not None:
            X_val, y_val = self.validation_data
            best_score = float('inf')
            for name, regr in self.base_regressors:
                val_predictions = regr.predict(X_val)
                score = mean_squared_error(y_val, val_predictions)
                if score < best_score:
                    best_score = score
                    self.final_estimator_ = regr

        # If no validation data provided, use the first loaded model as the final estimator
        if self.final_estimator_ is None:
            self.final_estimator_ = self.base_regressors[0][1]

        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self, 'final_estimator_')

        # Use the selected final estimator to make predictions
        return self.final_estimator_.predict(X)

    def get_selected_models(self):
        # Check if fit has been called
        return self.selected_model_files

# Example usage:
# Assuming you have a directory with trained model files and validation data (X_val, y_val):
# models_directory = 'path_to_your_models_directory'
# stacking_regressor_loader = GL__StackingRegressorLoader(models_dir=models_directory, validation_data=(X_val, y_val))
# stacking_regressor_loader.fit(X_train, y_train)  # This will load the models and select the final estimator
# predictions = stacking_regressor_loader.predict(X_test)
# selected_models = stacking_regressor_loader.get_selected_models()

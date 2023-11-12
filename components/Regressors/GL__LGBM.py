from sklearn.base import BaseEstimator, RegressorMixin
import lightgbm as lgb
import os

class GL__LGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.params = kwargs
        self.booster = None
        self.model_path = None  # Temporary path to store the LightGBM model

    def fit(self, X, y):
        lgb_train = lgb.Dataset(X, y)  # Temporary, so we don't store it
        self.booster = lgb.train(self.params, lgb_train)
        return self

    def predict(self, X):
        return self.booster.predict(X)

    # Additional methods for serialization:
    def save_model(self, path):
        self.model_path = path
        self.booster.save_model(path)

    def load_model(self, path):
        self.model_path = path
        self.booster = lgb.Booster(model_file=path)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Save the booster to a temporary file only if it exists
        if self.booster is not None:
            self.save_model("temp_lgb_model.txt")
            with open("temp_lgb_model.txt", "rb") as f:
                state["booster_"] = f.read()
            os.remove("temp_lgb_model.txt")  # Remove the file after reading
        else:
            state["booster_"] = None
        return state

    def __setstate__(self, state):
        # Load the booster from the saved state
        if state["booster_"] is not None:
            with open("temp_lgb_model.txt", "wb") as f:
                f.write(state["booster_"])
            self.load_model("temp_lgb_model.txt")
            os.remove("temp_lgb_model.txt")  # Remove the file after loading
        else:
            self.booster = None
        state.pop("booster_", None)  # Clean up the temporary entry
        self.__dict__.update(state)  # Restore the remaining state

# Example usage
# reg = GL__LGBMRegressor(num_leaves=31, learning_rate=0.05, n_estimators=20)
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test)

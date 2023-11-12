from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class Taxi_Outlier_Removal(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # nothing to fit

    def transform(self, X, y=None):
        # Assuming X is a DataFrame for easy boolean indexing
        df = pd.DataFrame(X, columns=['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag'])

        # Outlier removal
        df = df[(df.passenger_count > 0)]
        df = df[(df.pickup_longitude > -100)]
        df = df[(df.pickup_latitude < 50)]

        return df.values  # Return as numpy array
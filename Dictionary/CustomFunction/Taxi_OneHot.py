from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_encode=None):
        self.columns_to_encode = columns_to_encode

    def fit(self, X, y=None):
        # Store unique values for each column to ensure consistent one-hot encoding
        self.unique_values_ = {col: pd.Series(X[:, i]).unique() for i, col in enumerate(self.columns_to_encode)}
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=['id', 'vendor_id', 'pickup_datetime', 'dropoff_datetime', 'passenger_count',
                                      'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
                                      'store_and_fwd_flag'])

        for col in self.columns_to_encode:
            unique_values = df[col].nunique()

            # If only one unique value, set drop_first to False
            drop_first_value = True if unique_values > 1 else False
            # One-hot encode the column with drop_first=False to ensure all columns are returned
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first_value)
            # Drop the original column
            df.drop([col], axis=1, inplace=True)
            # Concatenate the one-hot encoded columns to df
            df = pd.concat([df, dummies], axis=1)


        return df.values

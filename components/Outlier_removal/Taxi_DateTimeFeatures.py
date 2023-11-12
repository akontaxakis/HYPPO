from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

def try_convert_to_float(arr):
    try:
        return arr.astype(float)
    except ValueError:
        # If conversion fails, identify problematic values
        for item in arr:
            try:
                float(item)
            except ValueError:
                print(f"Cannot convert '{item}' to float")
        return arr  # return original array

def ft_haversine_distance(lat1, lng1, lat2, lng2):
    lat1 = try_convert_to_float(lat1)
    lng1 = try_convert_to_float(lng1)
    lat2 = try_convert_to_float(lat2)
    lng2 = try_convert_to_float(lng2)

    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


# Direction computation
def ft_degree(lat1, lng1, lat2, lng2):

    lat1 = try_convert_to_float(lat1)
    lng1 = try_convert_to_float(lng1)
    lat2 = try_convert_to_float(lat2)
    lng2 = try_convert_to_float(lng2)

    AVG_EARTH_RADIUS = 6371  # km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Nothing to fit here
        return self

    def transform(self, X, y=None):

        df = pd.DataFrame(X, columns=['id', 'pickup_datetime', 'dropoff_datetime','passenger_count', 'pickup_latitude','pickup_longitude', 'dropoff_longitude', 'dropoff_latitude','store_and_fwd_flag','vendor_id'])
        # Datetime processing

        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        #df.drop(['dropoff_datetime'], axis=1, inplace=True)  # Drop as it's not in the test set

        # Extract datetime features
        df['month'] = df['pickup_datetime'].dt.month
        df['week'] = df['pickup_datetime'].dt.isocalendar().week
        df['weekday'] = df['pickup_datetime'].dt.weekday
        df['hour'] = df['pickup_datetime'].dt.hour
        df['minute'] = df['pickup_datetime'].dt.minute
        df['minute_oftheday'] = df['hour'] * 60 + df['minute']
        df.drop(['minute', 'pickup_datetime'], axis=1, inplace=True)
        # Add distance and direction features
        df['distance'] = ft_haversine_distance(df['pickup_latitude'].values,df['pickup_longitude'].values,df['dropoff_latitude'].values,df['dropoff_longitude'].values)
        df['direction'] = ft_degree(df['pickup_latitude'].values,df['pickup_longitude'].values,df['dropoff_latitude'].values,df['dropoff_longitude'].values)

        # Outlier removal
        #df = df[df['distance'] < 200]
        # Speed computation and outlier removal
        #df['speed'] = df['distance'] / df['trip_duration']
        #df = df[df['speed'] < 30]
        #df.drop(['speed'], axis=1, inplace=True)

        # Assuming "trip_duration" and "id" are not features
        #df.drop(["trip_duration", "id"], axis=1, inplace=True)
        df.drop(["vendor_id"], axis=1, inplace=True)
        df.drop(["id"], axis=1, inplace=True)
        df.drop(['dropoff_datetime'], axis=1, inplace=True)
        return df.values  # Return as numpy array

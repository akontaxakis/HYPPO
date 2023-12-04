
import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Dictionary.Outlier_removal.Taxi_DateTimeFeatures import CustomFeatureEngineer
from Dictionary.Outlier_removal.Taxi_OneHot import CustomOneHotEncoder
from Dictionary.Outlier_removal.Taxi_Outlier_Removal import Taxi_Outlier_Removal

if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")

    print(os.getcwd())
    categorical_features = ['store_and_fwd_flag', 'vendor_id']
    preprocessing_pipeline = Pipeline([
        ('OR', Taxi_Outlier_Removal()),
        ('OH', CustomOneHotEncoder(categorical_features)),
        ('FE', CustomFeatureEngineer()),
        ('SS', StandardScaler())
    ])

    clf1_loaded = joblib.load('taxi_models/X_SKCuSKCuSKGr2907')
    clf2_loaded = joblib.load('taxi_models/X_SKCuSKCuSKKN0663')
    clf3_loaded = joblib.load('taxi_models/X_SKCuSKCuSKRa2907')
    clf4_loaded = joblib.load('taxi_models/X_SKTaSKCuSKCuSKStSKKN3874')

    data = pd.read_csv('C:/Users/adoko/PycharmProjects/pythonProject1/datasets/taxi_train.csv')
    data['trip_duration'] = data['trip_duration'].replace(-1, 0)
    y = data['trip_duration'].values
    X = data.drop('trip_duration', axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessed_data = preprocessing_pipeline.fit_transform(X_train)
    preprocessed_test = preprocessing_pipeline.transform(X_test)
    predictions = clf1_loaded.predict(preprocessed_test)
    predictions = clf2_loaded.predict(preprocessed_test)
    y_train_aligned = y_train[:len(preprocessed_data)]

    votingC = VotingRegressor(estimators=[('rfc', clf1_loaded), ('extc', clf2_loaded)],
                               n_jobs=4)

    votingC = votingC.fit(preprocessed_data, y_train_aligned)
    predictions = votingC.predict(preprocessed_test)


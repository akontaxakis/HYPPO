import random

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor


def random_anomaly_pipeline():
    # Define preprocessing steps and anomaly detection algorithms
    preprocessors = [
        ('standard_scaler', StandardScaler()),
        ('minmax_scaler', MinMaxScaler()),
        ('robust_scaler', RobustScaler()),
        ('pca', PCA(n_components=random.choice([5, 10, 15]))),
    ]
    anomaly_detectors = [
        ('elliptic_envelope', EllipticEnvelope()),
        ('isolation_forest', IsolationForest()),
        ('one_class_svm', OneClassSVM())
    ]

    # Randomly select preprocessing steps and an anomaly detection algorithm
    num_preprocessors = random.randint(0, 4)
    pipeline_steps = random.sample(preprocessors, num_preprocessors)
    pipeline_steps.append(random.choice(anomaly_detectors))

    return Pipeline(pipeline_steps)

if __name__ == '__main__':

    sum = 0;
    data = load_breast_cancer()
    X, y = data.data, data.target
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Generate 100 random pipelines
    num_pipelines = 100
    random_pipelines = [random_anomaly_pipeline() for _ in range(num_pipelines)]

    # Define a normal class (0: malignant, 1: benign) and modify the labels
    normal_class = 1
    y_train_modified = np.where(y_train == normal_class, 1, -1)
    y_test_modified = np.where(y_test == normal_class, 1, -1)

    # Train and evaluate the pipelines
    for i, pipeline in enumerate(random_pipelines):
        sum = sum+(len(pipeline.steps))
        print(pipeline.steps)
        # If the pipeline ends with LocalOutlierFactor, we need to use fit_predict instead of fit
        if isinstance(pipeline.steps[-1][1], LocalOutlierFactor):
            y_pred_train = pipeline.fit_predict(X_train)
            y_pred_test = pipeline.predict(X_test)
        else:
            pipeline.fit(X_train[y_train_modified == 1])
            y_pred_test = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test_modified, y_pred_test)
        print(f"Pipeline {i + 1} accuracy: {accuracy:.4f}")
print(sum)

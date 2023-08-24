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

from skit_learn_components.SS_GPU import GPU__StandardScaler

if __name__ == '__main__':
    import time
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.covariance import EllipticEnvelope
    from skit_learn_components.GPU__PCA import GPU__PCA
    # Load the breast_cancer dataset and split it into training and test sets
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the pipeline
    pipeline_steps = [
        ('standard_scaler', GPU__StandardScaler()),
        ('pca', GPU__PCA(n_components=10)),
        ('elliptic_envelope', EllipticEnvelope())
    ]

    normal_class = 1
    y_train_modified = np.where(y_train == normal_class, 1, -1)
    y_test_modified = np.where(y_test == normal_class, 1, -1)

    pipeline = Pipeline(pipeline_steps)

    # Train the pipeline, measuring the execution time of each step
    for step_name, step in pipeline_steps:
        start_time = time.time()
        step.fit(X_train)
        if step_name == 'elliptic_envelope':
            step.fit(X_train[y_train_modified == 1])
            y_pred_test = pipeline.predict(X_test)
        else:
            step.fit(X_train)
            X_train = step.transform(X_train)
        end_time = time.time()
        print(f"{step_name} execution time: {end_time - start_time:.4f} seconds")

    # Evaluate the pipeline
    accuracy = accuracy_score(y_test_modified, y_pred_test)
    print(f"Pipeline accuracy: {accuracy:.4f}")

import time

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from memory_profiler import memory_usage
from sklearn.covariance import EllipticEnvelope
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

if __name__ == '__main__':
    # Load the breast_cancer dataset and split it into training and test sets
    G = nx.DiGraph()
    G.add_node("source")
    G.add_node("breast_cancer")
    start_time = time.time()
    data = load_breast_cancer()
    end_time = time.time()
    G.add_edge("source", "breast_cancer", weight=end_time - start_time +0.0001,  execution_time=end_time - start_time, memory_usage=0)
    mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)[0]
    start_time = time.time()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    end_time = time.time()
    mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)[0]
    G.add_node("X_test")
    #G.add_node("X_test")
    G.add_node("X_train")
    #G.add_node("y_test")
    G.add_edge("breast_cancer", "split", weight=end_time - start_time, execution_time=end_time - start_time, memory_usage=mem_usage_after - mem_usage_before)
    #G.add_edge("split", "X_train",weight=0.1, execution_time=0,
    #           memory_usage=0)
    G.add_edge("split", "X_test",weight=0.000001, execution_time=0,
               memory_usage=0)
    G.add_edge("split", "X_train",weight=0.000001, execution_time=0,
               memory_usage=0)
    #G.add_edge("split", "y_test",weight=0.1, execution_time=0,
    #           memory_usage=0)
    # Define the pipeline
    pipeline_steps = [
        ('gpu_standard_scaler', GPU__StandardScaler()),
        ('gpu_pca', GPU__PCA(n_components=10)),
        ('elliptic_envelope', EllipticEnvelope())
    ]

    normal_class = 1
    y_train_modified = np.where(y_train == normal_class, 1, -1)
    y_test_modified = np.where(y_test == normal_class, 1, -1)

    pipeline = Pipeline(pipeline_steps)

    # Train the pipeline, measuring the execution time and memory usage of each step
    transformed_X_train = X_train
    transformed_X_test = X_test

    previous_step = "X_train"
    for step_name, step in pipeline_steps:
        mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)[0]
        start_time = time.time()
        step.fit(transformed_X_train)
        if step_name == 'elliptic_envelope':
            step.fit(X_train[y_train_modified == 1])
            y_pred_test = pipeline.predict(X_test)
            end_time = time.time()
            mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)[0]
            mem_usage_diff = mem_usage_after - mem_usage_before
            execution_time = end_time - start_time
            G.add_node(step_name)
            if previous_step is not None:
                G.add_edge("X_test", "Combine", label=execution_time, weight=0.000001,
                           execution_time=execution_time, memory_usage=mem_usage_diff)
                G.add_edge(previous_step, "Combine", label=execution_time, weight=0.000001,
                           execution_time=execution_time, memory_usage=mem_usage_diff)
                G.add_edge("Combine", step_name, label=execution_time, weight=execution_time,
                           execution_time=execution_time, memory_usage=mem_usage_diff)
            previous_step = step_name
        else:
            step.fit(X_train)
            X_train = step.transform(X_train)
            end_time = time.time()
            mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)[0]
            mem_usage_diff = mem_usage_after - mem_usage_before
            execution_time = end_time - start_time
            G.add_node(step_name)
            if previous_step is not None:
                G.add_edge(previous_step, step_name,label=execution_time, weight=execution_time, execution_time=execution_time, memory_usage=mem_usage_diff)
            previous_step = step_name



    nx.write_graphml(G, "../python_playground/mygraph_3.graphml")
import time

import networkx as nx
from matplotlib import pyplot as plt
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from components.GPU__PCA import GPU__PCA
from components.SS_GPU import GPU__StandardScaler

if __name__ == '__main__':
    # Load the breast_cancer dataset and split it into training and test sets
    G = nx.DiGraph()
    G.add_node("source")
    G.add_node("breast_cancer")
    start_time = time.time()
    data = load_breast_cancer()
    end_time = time.time()
    G.add_edge("source", "breast_cancer", execution_time=end_time - start_time, memory_usage=0)
    mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)[0]
    start_time = time.time()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    end_time = time.time()
    mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)[0]
    G.add_node("X_train")
    G.add_node("X_test")
    G.add_node("y_train")
    G.add_node("y_test")
    G.add_edge("breast_cancer", "split", execution_time=end_time - start_time, memory_usage= mem_usage_after - mem_usage_before)
    G.add_edge("split", "X_train", execution_time=0,
               memory_usage=0)
    G.add_edge("split", "X_test", execution_time=0,
               memory_usage=0)
    G.add_edge("split", "y_train", execution_time=0,
               memory_usage=0)
    G.add_edge("split", "y_test", execution_time=0,
               memory_usage=0)
    # Define the pipeline
    pipeline_steps = [
        ('standard_scaler', GPU__StandardScaler()),
        ('pca', GPU__PCA(n_components=10)),
        ('svc', SVC())
    ]

    pipeline = Pipeline(pipeline_steps)

    # Train the pipeline, measuring the execution time and memory usage of each step
    transformed_X_train = X_train
    transformed_X_test = X_test
    for step_name, step in pipeline_steps[:-1]:
        mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)[0]
        start_time = time.time()
        step.fit(transformed_X_train)
        transformed_X_train = step.transform(transformed_X_train)
        transformed_X_test = step.transform(transformed_X_test)
        end_time = time.time()
        mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)[0]
        mem_usage_diff = mem_usage_after - mem_usage_before
        print(f"{step_name} execution time: {end_time - start_time:.4f} seconds")
        print(f"{step_name} memory usage: {mem_usage_diff:.4f} MiB")

        #G.add_node(step_name, step=step)
        #if previous_step is not None:
        #    G.add_edge(previous_step, step_name, execution_time=execution_time, memory_usage=memory_usage_diff)

    # Train and time the last step (SVC) separately
    step_name, step = pipeline_steps[-1]
    mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)[0]
    start_time = time.time()
    step.fit(transformed_X_train, y_train)
    end_time = time.time()
    mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)[0]
    mem_usage_diff = mem_usage_after - mem_usage_before
    print(f"{step_name} execution time: {end_time - start_time:.4f} seconds")
    print(f"{step_name} memory usage: {mem_usage_diff:.4f} MiB")

    # Evaluate the pipeline
    y_pred_test = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Pipeline accuracy: {accuracy:.4f}")


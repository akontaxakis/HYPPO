import os
import pickle
import time

import networkx as nx
import numpy as np
import pandas as pd
import pycuda
from memory_profiler import memory_usage
from sklearn import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from Components.parser.sub_parser import generate_pipeline, compute_pipeline_metrics, \
    compute_pipeline_metrics_training_ad, compute_pipeline_metrics_evaluation_ad, compute_pipeline_metrics_training, \
    compute_pipeline_metrics_evaluation, compute_pipeline_metrics_evaluation_helix, \
    compute_pipeline_metrics_training_helix


def generate_shared_graph(dataset, artifact_graph, uid, steps, N, task, X, y, mode, graph_dir='graphs/shared_graphs',
                          artifact_dir='artifacts'):
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    shared_graph_file = uid + "_shared_graph"
    shared_graph_path = os.path.join(graph_dir, f"{shared_graph_file}.plk")
    if os.path.exists(shared_graph_path):
        with open(shared_graph_path, 'rb') as f:
            print("load" + shared_graph_path)
            artifact_graph = pickle.load(f)
    # print(type(X))
    sum = 0
    artifacts = []
    if task == 'classifier':
        mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)[0]

        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        end_time = time.time()

        mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)[0]
        artifact_graph.add_node("X_train__")
        # G.add_node("X_test")
        artifact_graph.add_node("y_train__")
        # G.add_node("y_test")
        artifact_graph.add_edge(dataset, "split", weight=end_time - start_time,
                                memory_usage=mem_usage_after - mem_usage_before)
        artifact_graph.add_edge("split", "X_train__", weight=0.000001, execution_time=0,
                                memory_usage=0)
        # G.add_edge("split", "X_test",weight=0, execution_time=0,
        #           memory_usage=0)
        artifact_graph.add_edge("split", "y_train__", weight=0.000001, execution_time=0,
                                memory_usage=0)

        if mode == "sampling":
            start_time = time.time()

            sample_size = int(X_train.shape[0] * 0.1)

            # Generate a list of indices based on the length of your training set
            indices = np.arange(X_train.shape[0])
            # Randomly select indices for your sample
            sample_indices = np.random.choice(indices, size=sample_size, replace=False)

            # Use these indices to sample from X_train and y_train
            sample_X_train = X_train[sample_indices]
            sample_y_train = y_train[sample_indices]

            end_time = time.time()
            artifact_graph.add_edge("X_train__", "2sample_X_train__", weight=0.000001,
                                    execution_time=end_time - start_time,
                                    memory_usage=0)
            # G.add_edge("split", "X_test",weight=0, execution_time=0,
            #           memory_usage=0)
            artifact_graph.add_edge("y_train__", "2sample_y_train__", weight=0.000001,
                                    execution_time=end_time - start_time,
                                    memory_usage=0)

    for i in range(N):
        pipeline = generate_pipeline(steps, len(steps))
        print(pipeline)
        try:
            # Check if the pipeline has a classifier,
            has_evaluator = any(step_name == task for step_name, _ in pipeline.steps)
            if has_evaluator:
                if mode == "sampling":
                    pipeline.fit(sample_X_train, sample_y_train)
                    score = pipeline.score(X_test, y_test)
                    artifact_graph, artifacts = compute_pipeline_metrics(artifact_graph, pipeline, uid, sample_X_train,
                                                                         X_test, sample_y_train, y_test, artifacts,
                                                                         mode)
                else:
                    pipeline.fit(X_train, y_train)
                    score = pipeline.score(X_test, y_test)
                    artifact_graph, artifacts = compute_pipeline_metrics(artifact_graph, pipeline, uid, X_train, X_test,
                                                                         y_train, y_test, artifacts, mode)
                sum = sum + 1
            else:
                score = None


        # except TypeError:
        #   print("Oops!  Wrong Type.  Try again...")
        #   print(pipeline)
        # except ValueError:
        #    print("Oops!  That was no valid number.  Try again...")
        #    print(pipeline)
        except pycuda._driver.LogicError:
            print("Oops!  cuda")
            print(pipeline)
        # except InvalidArgumentError:
        #    print("Oops!  tensorflow")
        #    print(pipeline)
        # except AttributeError:
        #    print("Oops!  Attribute")
        #    print(pipeline)
    with open(shared_graph_path, 'wb') as f:
        print("save" + shared_graph_path)
        pickle.dump(artifact_graph, f)

    return artifact_graph, artifacts


def generate_shared_graph_m(dataset, artifact_graph, uid, steps, N, task, X, y, mode, graph_dir='graphs/shared_graphs',
                          artifact_dir='artifacts'):
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    shared_graph_file = uid + "_shared_graph"
    shared_graph_path = os.path.join(graph_dir, f"{shared_graph_file}.plk")
    if os.path.exists(shared_graph_path):
        with open(shared_graph_path, 'rb') as f:
            print("load" + shared_graph_path)
            artifact_graph = pickle.load(f)
    #print(type(X))
    sum = 0
    artifacts = []
    if task == 'classifier':

        start_time = time.time()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        end_time = time.time()
        mem_usage = memory_usage(lambda: train_test_split(X, y, test_size=0.2, random_state=42))
        artifact_graph.add_node("X_train__")
        # G.add_node("X_test")
        artifact_graph.add_node("y_train__")
        # G.add_node("y_test")
        artifact_graph.add_edge(dataset, "split", weight=end_time - start_time,execution_time=end_time - start_time,
                                memory_usage=max(mem_usage))
        artifact_graph.add_edge("split", "X_train__", weight=0.000001, execution_time=0,
                                memory_usage=0)
        # G.add_edge("split", "X_test",weight=0, execution_time=0,
        #           memory_usage=0)
        artifact_graph.add_edge("split", "y_train__", weight=0.000001, execution_time=0,
                                memory_usage=0)

        if mode == "sampling":
            mem_usage = memory_usage(lambda:sample(X_train, y_train, 0.1))
            start_time = time.time()
            sample_X_train, sample_y_train = sample(X_train, y_train, 0.1)
            end_time = time.time()
            artifact_graph.add_edge("X_train__", "2sample_X_train__", weight=end_time - start_time, execution_time=end_time - start_time,
                                memory_usage=max(mem_usage))
        # G.add_edge("split", "X_test",weight=0, execution_time=0,
        #           memory_usage=0)
            artifact_graph.add_edge("y_train__", "2sample_y_train__", weight=end_time - start_time, execution_time=end_time - start_time,
                                memory_usage=max(mem_usage))

    for i in range(N):
        pipeline = generate_pipeline(steps, len(steps))
        print(pipeline)
        try:
            # Check if the pipeline has a classifier,
            has_evaluator = any(step_name == task for step_name, _ in pipeline.steps)
            if has_evaluator:
                if mode == "sampling":
                    pipeline.fit(sample_X_train, sample_y_train)
                    score = pipeline.score(X_test, y_test)
                    artifact_graph, artifacts = compute_pipeline_metrics(artifact_graph, pipeline, uid, sample_X_train, X_test, sample_y_train, y_test, artifacts, mode)
                else:
                    pipeline.fit(X_train, y_train)
                    score = pipeline.score(X_test, y_test)
                    artifact_graph, artifacts = compute_pipeline_metrics(artifact_graph, pipeline, uid, X_train, X_test, y_train, y_test, artifacts, mode)
                sum = sum + 1
            else:
                score = None


        #except TypeError:
        #   print("Oops!  Wrong Type.  Try again...")
        #   print(pipeline)
        #except ValueError:
        #    print("Oops!  That was no valid number.  Try again...")
        #    print(pipeline)
        except pycuda._driver.LogicError:
            print("Oops!  cuda")
            print(pipeline)
        #except InvalidArgumentError:
        #    print("Oops!  tensorflow")
        #    print(pipeline)
        #except AttributeError:
        #    print("Oops!  Attribute")
        #    print(pipeline)
    with open(shared_graph_path, 'wb') as f:
        print("save" + shared_graph_path)
        pickle.dump(artifact_graph, f)

    return artifact_graph, artifacts

def sample(X_train, y_train, rate):
    sample_size = int(X_train.shape[0] * rate)
    # Generate a list of indices based on the length of your training set
    indices = np.arange(X_train.shape[0])
    # Randomly select indices for your sample
    sample_indices = np.random.choice(indices, size=sample_size, replace=False)
    # Use these indices to sample from X_train and y_train
    sample_X_train = X_train[sample_indices]
    sample_y_train = y_train[sample_indices]
    return sample_X_train, sample_y_train

def init_graph(dataset):
    # Load the Breast Cancer Wisconsin dataset
    cc = 0
    G = nx.DiGraph()
    G.add_node("source", type="source", size= 0, cc= cc)
    start_time = time.time()
    if dataset == "breast_cancer" :
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = np.random.rand(100000, 100)
        y = np.random.rand(100000)
        y = (y > 0.5).astype(int)
    elif dataset == "HIGGS":
        data = np.loadtxt('C:/Users/adoko/Downloads/HIGGS.csv', delimiter=',')
        # Extract and modify the first column based on your condition
        # (e.g., setting it to 0 or 1 if it's greater than 0.5)
        y = np.where(data[:, 0] > 0.5, 1, 0).astype(float)

        # Store the original first column in a separate array
        y = data[:, 0].copy()

        # Drop the first column from the data
        X = data[:, 1:]
        print(data.shape)
        print(data.shape)
    elif dataset == "TAXI":
        data = pd.read_csv('C:/Users/adoko/PycharmProjects/pythonProject1/datasets/taxi_train.csv')
        data['trip_duration'] = data['trip_duration'].replace(-1, 0)
        y = data['trip_duration'].values
        X = data.drop('trip_duration', axis=1).values
        test = pd.read_csv('C:/Users/adoko/PycharmProjects/pythonProject1/datasets/taxi_test.csv')
    else:
        data = pd.read_csv('C:/Users/adoko/Υπολογιστής/BBC.csv')
        data['target'] = data['target'].replace(-1, 0)
        y = data['target'].values
        X = data.drop('target', axis=1).values

    # data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    end_time = time.time()
    cc =end_time - start_time
    G.add_node(dataset, type="raw", size=X.size * X.itemsize, cc = cc, frequency = 1)
    platforms = []
    platforms.append("python")
    G.add_edge("source", dataset,type="load", weight=end_time - start_time + 0.000001, execution_time=end_time - start_time,
               memory_usage=0, platform = platforms)
    return X, y, G, cc

def execute_pipeline(dataset, artifact_graph, uid, steps,mode,cc,X_train, y_train,X_test, y_test):
    #artifact_graph, shared_graph_path = extract_artifact_graph(artifact_graph, uid)
    sum = 0
    artifacts = []
    pipeline = steps
    ##pipeline = generate_pipeline(steps, len(steps))
    print(pipeline)
    try:
        new_pipeline = clone(pipeline)
        cc1 = cc
        artifact_graph, artifacts, new_pipeline = compute_pipeline_metrics_training(artifact_graph, new_pipeline, uid,X_train, y_train, artifacts, mode,cc1)
        artifact_graph, artifacts, request = compute_pipeline_metrics_evaluation(artifact_graph, new_pipeline, uid,X_test, y_test, artifacts)

        sum = sum + 1
        #except TypeError:
        #   print("Oops!  Wrong Type.  Try again...")
        #   print(pipeline)
        #except ValueError:
        #    print("Oops!  That was no valid number.  Try again...")
        #    print(pipeline)
    except pycuda._driver.LogicError:
        print("Oops!  cuda")
        print(pipeline)
        #except InvalidArgumentError:
        #    print("Oops!  tensorflow")
        #    print(pipeline)
        #except AttributeError:
        #    print("Oops!  Attribute")
        #    print(pipeline)
   # with open(shared_graph_path, 'wb') as f:
   #     print("save" + shared_graph_path)
   #     pickle.dump(artifact_graph, f)

    return artifact_graph, artifacts,request

def execute_pipeline_helix(dataset, artifact_graph, uid, steps,mode,cc,X_train, y_train,X_test, y_test, budget):

    artifacts = []
    pipeline = steps
    ##pipeline = generate_pipeline(steps, len(steps))
    print(pipeline)
    try:
        new_pipeline = clone(pipeline)
        cc1 = cc
        artifact_graph, artifacts, new_pipeline, materialized_artifacts, budget = compute_pipeline_metrics_training_helix(artifact_graph, new_pipeline, uid, X_train, y_train, artifacts, mode, cc1, budget)
        artifact_graph, artifacts, request, materialized_artifacts = compute_pipeline_metrics_evaluation_helix(artifact_graph, new_pipeline, uid, X_test, y_test, artifacts, materialized_artifacts, budget)

        #except TypeError:
        #   print("Oops!  Wrong Type.  Try again...")
        #   print(pipeline)
        #except ValueError:
        #    print("Oops!  That was no valid number.  Try again...")
        #    print(pipeline)
    except pycuda._driver.LogicError:
        print("Oops!  cuda")
        print(pipeline)
        #except InvalidArgumentError:
        #    print("Oops!  tensorflow")
        #    print(pipeline)
        #except AttributeError:
        #    print("Oops!  Attribute")
        #    print(pipeline)
   # with open(shared_graph_path, 'wb') as f:
   #     print("save" + shared_graph_path)
   #     pickle.dump(artifact_graph, f)

    return artifact_graph, artifacts,request,materialized_artifacts

def execute_pipeline_ad(dataset, artifact_graph, uid, steps,mode,cc,X_train, y_train,X_test, y_test ):
    #artifact_graph, shared_graph_path = extract_artifact_graph(artifact_graph, uid)
    sum = 0
    artifacts = []
    pipeline = steps
    ##pipeline = generate_pipeline(steps, len(steps))
    print(pipeline)

    new_pipeline = clone(pipeline)
    cc1 = cc
    artifact_graph, artifacts, new_pipeline, selected_models = compute_pipeline_metrics_training_ad(artifact_graph, new_pipeline, uid, X_train, y_train, artifacts, mode,cc1)
    artifact_graph, artifacts, request = compute_pipeline_metrics_evaluation_ad(artifact_graph, new_pipeline, uid, X_test, y_test, artifacts)

    return artifact_graph, artifacts, request

def split_data(X, artifact_graph, dataset, mode, y, cc):
    platforms = []
    platforms.append("python")
    start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    end_time = time.time()
    mem_usage = [0,0]# memory_usage(lambda: train_test_split(X, y, test_size=0.2, random_state=42))
    # G.add_node("y_test")
    step_time = end_time - start_time
    cc = cc + step_time

    artifact_graph.add_node("X_train__", type="training", size=X_train.__sizeof__(),cc=cc,frequency = 1)
    artifact_graph.add_node("X_test__", type="test", size=X_test.__sizeof__(),cc=cc,frequency = 1)
    artifact_graph.add_node("split", type="split", size=0, cc=0,frequency = 1)

    artifact_graph.add_edge(dataset, "split",type="split", weight=step_time, execution_time=step_time,
                            memory_usage=max(mem_usage),platform = platforms)
    artifact_graph.add_edge("split", "X_train__",type="split", weight=0.000001, execution_time=0,
                            memory_usage=0,platform = platforms)
    # G.add_edge("split", "X_test",weight=0, execution_time=0,
    #           memory_usage=0)
    artifact_graph.add_edge("split", "X_test__",type="split", weight=0.000001, execution_time=0,
                            memory_usage=0,platform = platforms)
    if mode == "sampling":
        mem_usage = [0,0]#memory_usage(lambda: sample(X_train, y_train, 0.1))
        start_time = time.time()
        X_train, y_train = sample(X_train, y_train, 0.1)
        end_time = time.time()
        step_time = end_time - start_time
        artifact_graph.add_edge("X_train__", "2sample_X_train__", weight=end_time - start_time,
                                execution_time=end_time - start_time,
                                memory_usage=max(mem_usage),platform = platforms)
        # G.add_edge("split", "X_test",weight=0, execution_time=0,
        #           memory_usage=0)
        artifact_graph.add_edge("X_test__", "2sample_X_test__", weight=end_time - start_time,
                                execution_time=end_time - start_time,
                                memory_usage=max(mem_usage),platform = platforms)

    return X_test, X_train, y_test, y_train,cc




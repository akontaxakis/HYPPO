import os
import pickle
import time

import numpy as np
import pandas as pd
import pycuda
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from libs.Pipelines_Library import extract_first_two_chars, generate_pipeline, compute_pipeline_metrics


def load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs", mode="eq_"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + mode + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    return artifact_graph


def store_EDGES_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs"):
    file_name = uid + "_EDGES_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.txt")
    with open(ag_path, "w") as outfile:
        # Iterate over edges and write to file
        for u, v, data in artifact_graph.edges(data=True):
            cost = data['weight']
            outfile.write(f'"{u}","{v}",{cost}\n')
    return artifact_graph


def store_or_load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    else:
        with open(ag_path, 'wb') as f:
            pickle.dump(artifact_graph, f)

    file_name = uid + "_EDGES_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.txt")

    with open(ag_path, "w") as outfile:
        # Iterate over edges and write to file
        for u, v, data in artifact_graph.edges(data=True):
            cost = data['cost']
            outfile.write(f'"{u}","{v}",{cost}\n')
    return artifact_graph


def create_artifact_graph(artifacts):
    G = nx.DiGraph()
    for i, (step_name, artifact) in enumerate(artifacts.items()):
        G.add_node(step_name, artifact=artifact)
        if i > 0:
            prev_step_name = list(artifacts.keys())[i - 1]
            G.add_edge(prev_step_name, step_name)

    return G


def plot_artifact_graph(G, uid, type):
    plt.figure(figsize=(20, 18))
    pos = nx.drawing.layout.spring_layout(G, seed=620, scale=4)
    nx.draw(G, pos, with_labels=True, node_size=120, node_color="skyblue", font_size=5)
    folder_path = "plots/"
    file_path = os.path.join(folder_path, uid + "_" + type + "_plot.pdf")
    plt.savefig(file_path)
    plt.show()


def load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs", mode="eq_"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + mode + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    return artifact_graph


def add_load_tasks_to_the_graph(shared_artifact_graph, artifacts, artifact_dir='artifacts'):
    os.makedirs(artifact_dir, exist_ok=True)
    for artifact in artifacts:
        artifact_path = os.path.join(artifact_dir, f"{artifact}.pkl")
        start_time = time.time()
        with open(artifact_path, 'rb') as f:
            data = pickle.load(f)
            data1 = data
        f.close()
        loading_time = time.time() - start_time
        shared_artifact_graph.add_edge("source", artifact, weight=loading_time,
                                       execution_time=loading_time,
                                       memory_usage=0)

    return shared_artifact_graph


def create_equivalent_graph(uid, artifact_graph_2, metrics_dir='metrics'):
    artifact_graph = artifact_graph_2.copy()
    os.makedirs(metrics_dir, exist_ok=True)
    models_and_scores = []
    file_name_2 = uid + "_scores"
    scores_path = os.path.join(metrics_dir, f"{file_name_2}.txt")
    with open(scores_path, 'r') as file:
        for line in file:
            if (len(line.split('","')) > 1):
                model, score, eq_model, execution_time = line.split('","')
                new_model = (extract_first_two_chars(model.replace('"', '')), eq_model)
                if (not models_and_scores.__contains__(new_model)):
                    models_and_scores.append(new_model)

    for i in range(len(models_and_scores)):
        node_name = models_and_scores[i][1]
        if node_name not in artifact_graph.nodes:
            artifact_graph.add_node(node_name)
        if models_and_scores[i][0] in artifact_graph.nodes:
            artifact_graph = merge_nodes_2(artifact_graph, node_name, models_and_scores[i][0])

    artifact_graph = merge_GPU_nodes(artifact_graph)
    return artifact_graph


def merge_GPU_nodes(artifact_graph):
    nodes_to_remove = []
    nodes = artifact_graph.nodes()
    modified_graph = artifact_graph.copy()
    for node in nodes:
        if "GP" in node:
            modified_node = node.replace("GP", "")
            if modified_node in nodes:
                # Create new node with modified label (without "GPU")
                # if modified_node in artifact_graph.nodes():
                print("current")
                print(artifact_graph.in_edges(node, data=True))
                print(artifact_graph.in_edges(modified_node, data=True))
                # artifact_graph = nx.contracted_nodes(artifact_graph, modified_node, node)
                modified_graph = merge_nodes_2(modified_graph, modified_node, node)
                print("modified")
                print(modified_graph.in_edges(modified_node, data=True))

    return modified_graph


import networkx as nx


def merge_nodes(G: object, node1: object, node2: object,nodes_to_remove) -> object:
    # Combine the neighbors of both nodes
    out_neighbors = set(G.successors(node1)).union(G.successors(node2))
    in_neighbors = set(G.predecessors(node1)).union(G.predecessors(node2))

    # Iterate through the out-neighbors, adding edges between node1 and the neighbors
    for neighbor in out_neighbors:
        if neighbor == node1 or neighbor == node2:
            continue

        if G.has_edge(node1, neighbor) and G.has_edge(node2, neighbor):
            weight = min(G[node1][neighbor]['weight'], G[node2][neighbor]['weight'])
        elif G.has_edge(node1, neighbor):
            weight = G[node1][neighbor]['weight']
        else:
            weight = G[node2][neighbor]['weight']

        G.add_edge(node1, neighbor, weight=weight)
    # Iterate through the in-neighbors, adding edges between the neighbors and node1
    for neighbor in in_neighbors:
        if neighbor == node1 or neighbor == node2:
            continue

        if G.has_edge(neighbor, node1) and G.has_edge(neighbor, node2):
            weight = min(G[neighbor][node1]['weight'], G[neighbor][node2]['weight'])
        elif G.has_edge(neighbor, node1):
            weight = G[neighbor][node1]['weight']
        else:
            weight = G[neighbor][node2]['weight']

        G.add_edge(neighbor, node1, weight=weight)

    # Remove node2 from the graph
    nodes_to_remove.append(node2)
    #G.remove_node(node2)
    return G, nodes_to_remove

def merge_nodes_2(G: object, node1: object, node2: object) -> object:
    # Combine the neighbors of both nodes
    out_neighbors = set(G.successors(node1)).union(G.successors(node2))
    in_neighbors = set(G.predecessors(node1)).union(G.predecessors(node2))

    # Iterate through the out-neighbors, adding edges between node1 and the neighbors
    for neighbor in out_neighbors:
        if neighbor == node1 or neighbor == node2:
            continue

        if G.has_edge(node1, neighbor) and G.has_edge(node2, neighbor):
            weight = min(G[node1][neighbor]['weight'], G[node2][neighbor]['weight'])
        elif G.has_edge(node1, neighbor):
            weight = G[node1][neighbor]['weight']
        else:
            weight = G[node2][neighbor]['weight']

        G.add_edge(node1, neighbor, weight=weight)
    # Iterate through the in-neighbors, adding edges between the neighbors and node1
    for neighbor in in_neighbors:
        if neighbor == node1 or neighbor == node2:
            continue

        if G.has_edge(neighbor, node1) and G.has_edge(neighbor, node2):
            weight = min(G[neighbor][node1]['weight'], G[neighbor][node2]['weight'])
        elif G.has_edge(neighbor, node1):
            weight = G[neighbor][node1]['weight']
        else:
            weight = G[neighbor][node2]['weight']

        G.add_edge(neighbor, node1, weight=weight)

    # Remove node2 from the graph
    G.remove_node(node2)
    return G

def change_node_name(G, old_name, new_name):
    # Copy the attributes of the old node to the new node
    attributes = G.nodes[old_name]
    G.add_node(new_name, **attributes)

    # Iterate through the edges of the old node and add equivalent edges with the new node
    for neighbor, edge_attrs in G[old_name].items():
        G.add_edge(new_name, neighbor, **edge_attrs)

    # Remove the old node from the graph
    G.remove_node(old_name)


def create_equivalent_graph_2(steps, uid, artifact_graph, metrics_dir='metrics'):
    os.makedirs(metrics_dir, exist_ok=True)
    models_and_scores = []
    file_name_2 = uid + "_scores"
    scores_path = os.path.join(metrics_dir, f"{file_name_2}.txt")
    print(scores_path)
    with open(scores_path, 'r') as file:
        for line in file:
            if (len(line.split('","')) > 1):
                model, score, eq_model, execution_time = line.split('","')
                new_model = (extract_first_two_chars(model.replace('"', '')), score)
                if (not models_and_scores.__contains__(new_model)):
                    models_and_scores.append(new_model)

    for i in range(len(models_and_scores)):
        rounded_score = round(float(models_and_scores[i][1].strip('" \n')), 3)
        node_name = str(models_and_scores[i][0][-2:]) + "_" + str(rounded_score)
        artifact_graph.add_node(node_name)
        if models_and_scores[i][0] in artifact_graph.nodes:
            artifact_graph = nx.contracted_nodes(artifact_graph, node_name, models_and_scores[i][0])

    artifact_graph = merge_GPU_nodes(artifact_graph)
    return artifact_graph

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
            rus = RandomUnderSampler()
            #sample_X_train, sample_y_train = rus.fit_resample(X_train, y_train)

            sample_size = int(X_train.shape[0] * 0.1)

            # Generate a list of indices based on the length of your training set
            indices = np.arange(X_train.shape[0])
            # Randomly select indices for your sample
            sample_indices = np.random.choice(indices, size=sample_size, replace=False)

            # Use these indices to sample from X_train and y_train
            sample_X_train = X_train.iloc[sample_indices]
            sample_y_train = y_train.iloc[sample_indices]



            end_time = time.time()
            artifact_graph.add_edge("X_train__", "sample_X_train__", weight=0.000001, execution_time=end_time - start_time,
                                memory_usage=0)
        # G.add_edge("split", "X_test",weight=0, execution_time=0,
        #           memory_usage=0)
            artifact_graph.add_edge("y_train__", "sample_y_train__", weight=0.000001, execution_time=end_time - start_time,
                                memory_usage=0)

    for i in range(N):
        pipeline = generate_pipeline(steps, len(steps))
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


        except TypeError:
           print("Oops!  Wrong Type.  Try again...")
           print(pipeline)
        except ValueError:
            print("Oops!  That was no valid number.  Try again...")
            print(pipeline)
        except pycuda._driver.LogicError:
            print("Oops!  cuda")
            print(pipeline)
        except AttributeError:
            print("Oops!  Attribute")
            print(pipeline)
    with open(shared_graph_path, 'wb') as f:
        print("save" + shared_graph_path)
        pickle.dump(artifact_graph, f)

    return artifact_graph, artifacts


def init_graph(dataset):
    # Load the Breast Cancer Wisconsin dataset
    G = nx.DiGraph()
    G.add_node("source")
    G.add_node(dataset)
    start_time = time.time()
    if(dataset=="breast_cancer"):
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        val = pd.read_csv('C:/Users/adoko/Υπολογιστής/BBC.csv')
        val['target'] = val['target'].replace(-1, 0)
        y = val['target']
        X = val.drop('target', axis=1)
    # data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

    end_time = time.time()
    G.add_edge("source", dataset, weight=end_time - start_time + 0.000001, execution_time=end_time - start_time,
               memory_usage=0)
    return X, y, G

# This is a sample Python script.
# Press the green button in the gutter to run the script.
# Generate a random pipeline with 1 to 5 steps
import time

import networkx as nx
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from generator.steps_examples import simple_eq_steps, simple_eq_steps_1, ad_eq_steps, simple_eq_steps_bigger
from libs.Pipelines_Library import generate_pipeline, compute_pipeline_metrics
from libs.artifact_graph_lib import plot_artifact_graph, add_load_tasks_to_the_graph, create_equivalent_graph, \
    generate_shared_graph, init_graph, store_EDGES_artifact_graph

if __name__ == '__main__':
    import os

    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    # uid = str(uuid.uuid1())[:8]
    uid = "12350"
    #uid = "12346"
    steps =simple_eq_steps
    number_of_steps = len(steps)

    X, y, artifact_graph = init_graph()

    # Generate and execute 10 pipelines
    shared_artifact_graph, artifacts = generate_shared_graph(artifact_graph, uid, steps, 20, 'classifier', X, y)
    plot_artifact_graph(shared_artifact_graph, uid, "shared")
    print("")

    limited_shared_graph = add_load_tasks_to_the_graph(shared_artifact_graph, artifacts)
    plot_artifact_graph(limited_shared_graph, uid, "limited")

    equivalent_graph = create_equivalent_graph(steps, uid, limited_shared_graph, metrics_dir='metrics')
    plot_artifact_graph(equivalent_graph, uid, "equivalent")
    # store_or_load_artifact_graph(equivalent_graph, sum, uid, 'eq_classifier', 'breast_cancer')
    store_EDGES_artifact_graph(limited_shared_graph, 100, uid, 'lt_classifier', 'breast_cancer', graph_dir="graphs")
    store_EDGES_artifact_graph(equivalent_graph, 100, uid, 'eq_classifier', 'breast_cancer', graph_dir="graphs")
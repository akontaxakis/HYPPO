import random

import networkx as nx

from generator.pipelines_steps.first_set import GPU_steps_CL, TF_steps_CL, TR_steps_CL, SK_steps_CL, \
    GPU_steps_CN, TF_steps_CN, SK_steps_CN, TR_steps_CN
from libs.artifact_graph_lib import init_graph, generate_shared_graph, add_load_tasks_to_the_graph, plot_artifact_graph, \
    create_equivalent_graph, store_EDGES_artifact_graph, generate_shared_graph_m


def get_folder_size(folder_path):
    total_size = 0
    for path, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(path, file)
            total_size += os.path.getsize(file_path)
    return total_size


if __name__ == '__main__':
    import os
    dataset = "commercial"
    #dataset = "breast_cancer"
    uid = "CN2"
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")


    X, y, artifact_graph = init_graph(dataset)
    i = 0;


    artifact_graph, artifacts = generate_shared_graph_m(dataset, artifact_graph, uid, GPU_steps_CN, 1, 'classifier', X, y,"no_sampling")
    artifact_graph, artifacts = generate_shared_graph_m(dataset, artifact_graph, uid, SK_steps_CN, 1, 'classifier', X, y,"no_sampling")
    artifact_graph, artifacts = generate_shared_graph_m(dataset, artifact_graph, uid, TR_steps_CN, 1, 'classifier', X, y,"no_sampling")
    artifact_graph, artifacts = generate_shared_graph_m(dataset, artifact_graph, uid, TF_steps_CN, 1, 'classifier', X, y, "no_sampling")
    plot_artifact_graph(artifact_graph, uid, "shared")
    artifacts = []
    limited_shared_graph = add_load_tasks_to_the_graph(artifact_graph, artifacts)
    plot_artifact_graph(limited_shared_graph, uid, "limited")
    print(type(limited_shared_graph))
    equivalent_graph = create_equivalent_graph(uid, limited_shared_graph, metrics_dir='metrics')
    plot_artifact_graph(equivalent_graph, uid, "equivalent")
    # store_or_load_artifact_graph(equivalent_graph, sum, uid, 'eq_classifier', 'breast_cancer')
    store_EDGES_artifact_graph(limited_shared_graph, 100, uid, 'lt_classifier', 'breast_cancer', graph_dir="graphs")
    store_EDGES_artifact_graph(equivalent_graph, 100, uid, 'eq_classifier', 'breast_cancer', graph_dir="graphs")
    nx.write_graphml(equivalent_graph, 'eq.graphml')
    nx.write_graphml(limited_shared_graph, 'lt.graphml')
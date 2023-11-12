import random

import networkx as nx

from generator.steps_examples import AG_3_steps, person_1_steps, person_3_steps, person_2_steps, \
    steps_with_sampling, all_steps
from libs.artifact_graph_lib import init_graph, generate_shared_graph, add_load_tasks_to_the_graph, plot_artifact_graph, \
    create_equivalent_graph, store_EDGES_artifact_graph


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
    uid = "playground"
    uid = "AG_3"
    uid = "playground_commercial_AG_3"
    uid = "playground_commercial_4"
    uid = "all_in_one"
    uid = "test_sampling"
    uid = "sampling_50"
    uid = "sampling_200_sharing"
    uid = "test_size_2"
    uid = "all_steps"
    uid = "all_steps_2"
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    # uid = str(uuid.uuid1())[:8]
    # uid = "12346"
    data_size = []
    models_size = []
    #steps_choices = [person_1_steps,person_2_steps]
    steps_choices = [all_steps]
    X, y, artifact_graph = init_graph(dataset)
    i = 0;
    while i < 1:

        steps = random.choice(steps_choices)
        number_of_steps = len(steps)
        artifact_graph, artifacts = generate_shared_graph(dataset, artifact_graph, uid, steps, 30, 'classifier', X, y, "sampling")
        i = i + 1


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
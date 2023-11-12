import random

import networkx as nx

from generator.steps_examples import AG_3_steps, person_1_steps, person_3_steps, person_2_steps
from libs.artifact_graph_lib import init_graph, generate_shared_graph, add_load_tasks_to_the_graph, plot_artifact_graph, \
    create_equivalent_graph, store_EDGES_artifact_graph

if __name__ == '__main__':
    import os

    X, y, artifact_graph = init_graph()
    uid = "playground"
    uid = "AG_3"
    uid = "collaboration_2"
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    # uid = str(uuid.uuid1())[:8]
    # uid = "12346"
    steps_choices = [person_1_steps,person_2_steps,person_3_steps]
    # Generate and execute 10 pipelines
    i = 0
    while i < 1:
        steps = random.choice(steps_choices)
        number_of_steps = len(steps)
        artifact_graph, artifacts = generate_shared_graph(artifact_graph, uid, steps, 10, 'classifier', X, y)
        print("" + str(i))
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
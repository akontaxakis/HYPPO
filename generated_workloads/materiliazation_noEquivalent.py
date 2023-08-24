import random

import networkx as nx

from generated_workloads.pipelines_steps.first_set import SK_steps_CN, TF_steps_CN, TR_steps_CN, GPU_steps_CL, \
    SK_steps_CN_2, SK_steps_CN_1
from libs.artifact_graph_lib import init_graph, add_load_tasks_to_the_graph, plot_artifact_graph, \
    store_EDGES_artifact_graph, execute_pipeline, rank_based_materializer, new_edges, extract_nodes_and_edges
import heapq


def store_diff():
    os.makedirs('iterations', exist_ok=True)
    with open('iterations/' + uid + '_iterations_diff_' + str(iteration) + '.txt', 'w') as f:
        f.write(str(iteration) + ",")
        for node in required_nodes:
            f.write(str(node) + ",")
        f.write(str(extra_cost))
    print(required_nodes)


if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    Trails = [SK_steps_CN_1, TR_steps_CN]
    #dataset = "commercial"
    dataset = "breast_cancer"
    uid = "s5yn"
    X, y, raw_artifact_graph, cc = init_graph(dataset)
    iteration = 0

    artifact_graph_0, artifacts = execute_pipeline(dataset, raw_artifact_graph.copy(), uid, Trails[0], X, y, "no_sampling", cc)
    extract_nodes_and_edges(artifact_graph_0, uid, "shared", iteration)
    artifact_graph_1, artifacts = execute_pipeline(dataset, raw_artifact_graph.copy(), uid, Trails[1], X, y, "no_sampling", cc)
    iteration = iteration + 1
    required_nodes, extra_cost = new_edges(artifact_graph_0,artifact_graph_1)
    store_diff()

    Budget = raw_artifact_graph.nodes[dataset]['size']
    materialized_artifacts_0 = rank_based_materializer(artifact_graph_0, Budget)

    #print(materialized_artifacts)

    limited_shared_graph = add_load_tasks_to_the_graph(artifact_graph_0, materialized_artifacts_0)
    extract_nodes_and_edges(limited_shared_graph, uid,"limited", iteration)
    extract_nodes_and_edges(artifact_graph_1,uid,"shared",iteration)


    #plot_artifact_graph(limited_shared_graph, uid, "limited")
    #plot_artifact_graph(artifact_graph, uid, "shared")
    # store_or_load_artifact_graph(equivalent_graph, sum, uid, 'eq_classifier', 'breast_cancer')
    #store_EDGES_artifact_graph(limited_shared_graph, 100, uid, 'lt_classifier', 'breast_cancer', graph_dir="graphs")

   #extract_nodes_and_edges()

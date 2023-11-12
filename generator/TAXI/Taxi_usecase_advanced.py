import pickle
import random
from itertools import product

import joblib
import networkx as nx
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Example.user_iterations import collab_HIGGS_all_operators, collab_TAXI_all_operators, \
    collab_TAXI_all_operators_advance
from components.Outlier_removal.Taxi_DateTimeFeatures import CustomFeatureEngineer
from components.Outlier_removal.Taxi_OneHot import CustomOneHotEncoder
from components.Outlier_removal.Taxi_Outlier_Removal import Taxi_Outlier_Removal
from libs.artifact_graph_lib import init_graph, add_load_tasks_to_the_graph, execute_pipeline, rank_based_materializer, \
    new_edges, extract_nodes_and_edges, \
    split_data, create_equivalent_graph, new_eq_edges, create_equivalent_graph_without_fit, graphviz_draw, \
    graphviz_draw_with_requests, graphviz_draw_with_requests_and_new_tasks, execute_pipeline_ad
from libs.logical_pipeline_generator import logical_to_physical_random


def store_diff(required_nodes, extra_cost, request, uid):
    os.makedirs('iterations_diff', exist_ok=True)
    with open('iterations_diff/' + uid + '_iterations_diff_' + str(iteration) + '.txt', 'w') as f:
        for node in required_nodes:
            f.write(str(node) + ",")
        f.write(str(extra_cost) + ",")
        f.write(str(request))
    print(required_nodes)


def edge_match(e1, e2):
    return set(e2['platform']).issubset(set(e1['platform']))


def generate_configurations(all_steps):
    step_names = [step[0].split(".")[1] if '.' in step[0] else step[0] for step in all_steps]
    step_options = [step[1] for step in all_steps]
    for values in product(*step_options):
        yield [(name, [option]) for name, option in zip(step_names, values)]  # generate tuples




if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    dataset = "TAXI"
    uid = "TAXI_graph_100_ad"
    iteration = 10
    k = 100
    N = 10

    pickle_path = f"{uid}_{iteration}_advance.gpickle"
    with open(pickle_path, 'rb') as file:
        shared_graph_raw = pickle.load(file)
    uid = "TAXI_ADVANCED"
    iteration =0
    X, y, raw_artifact_graph, cc = init_graph(dataset)
    X_test, X_train, y_test, y_train, cc = split_data(X, raw_artifact_graph, dataset, "no_sampling", y, cc)
    dataset_size = raw_artifact_graph.nodes[dataset]['size']

    #dataset = "breast_cancer"
    #uid = "TAXI_10_200_f5"

    operators_dict = {key: value for key, value in collab_TAXI_all_operators_advance}
    #logical_pipelines_pool = "OR|OH|FE|;SS|RF();GBR();LGBM()|MSE"

    logical_pipelines_pool = "OR|OH|FE|SI|SS|VR(2);VR(3);VR(4);VR(5);SE(2);SE(3);SE(4);SE(5)|MSE;MAE;MPE"


    dataset_size = shared_graph_raw.nodes[dataset]['size']
    print(dataset_size)

    Budget = [0,dataset_size / 100000, dataset_size / 10000, dataset_size / 1000, dataset_size / 100, dataset_size / 10, dataset_size, dataset_size * 10, dataset_size * 100]
    #Budget = [dataset_size / 10, dataset_size, dataset_size * 10, dataset_size * 100,  dataset_size * 1000]

    loading_speed = 566255240
    for i in range(N):

        Trails = logical_to_physical_random(logical_pipelines_pool,operators_dict, k)
        print(Trails)
        sh_previous_graph = shared_graph_raw
        budget_it = 0
        iteration = 0
        for trial in Trails:
            #######################--SHARED GRAPH--#########################
            execution_graph, artifacts, request = execute_pipeline_ad(dataset, raw_artifact_graph.copy(), uid, trial,
                                                                   "no_sampling", cc, X_train, y_train, X_test, y_test)
            shared_artifact_graph = nx.compose(execution_graph, sh_previous_graph)
            extract_nodes_and_edges(shared_artifact_graph, uid+ "_" + str(i), "shared", iteration)

            ####################--Update Graphs for next iteration--#########
            budget_it = 0
            for b in Budget:
                sh_previous_graph_2 = sh_previous_graph.copy()

                ######################--LIMITED GRAPH--#########################
                limited_required_nodes, extra_cost_1, new_tasks = new_edges(sh_previous_graph, execution_graph)
                store_diff(limited_required_nodes, extra_cost_1, request, uid + "_" + str(budget_it) + "_" + str(i))
                materialized_artifacts_0 = rank_based_materializer(sh_previous_graph, b)
                limited_shared_graph = add_load_tasks_to_the_graph(sh_previous_graph, materialized_artifacts_0)
                extract_nodes_and_edges(limited_shared_graph, uid + "_" + str(budget_it) + "_" + str(i), "limited", iteration)
                #graphviz_draw_with_requests(limited_shared_graph, "lt", limited_required_nodes)
                #graphviz_draw_with_requests_and_new_tasks(limited_shared_graph, "lt", limited_required_nodes, new_tasks)
                ######################--EQUIVALENT GRAPH--######################

                # equivalent_graph, required_nodes, extra_cost = create_equivalent_graph_2(equivalent_graph,execution_graph.copy())
                equivalent_graph = create_equivalent_graph_without_fit(sh_previous_graph_2)
                required_nodes, extra_cost_2,new_tasks = new_eq_edges(execution_graph, equivalent_graph,"no_fit")
                store_diff(required_nodes, extra_cost_2, request, uid + "_eq_" + str(budget_it)+ "_" + str(i))
                materialized_artifacts_1 = rank_based_materializer(equivalent_graph, b)
                equivalent_graph = add_load_tasks_to_the_graph(equivalent_graph, materialized_artifacts_1)
                extract_nodes_and_edges(equivalent_graph, uid + "_" + str(budget_it)+ "_" + str(i), "equivalent", iteration)
                #graphviz_draw_with_requests(equivalent_graph, "eq", required_nodes)
                #graphviz_draw_with_requests_and_new_tasks(equivalent_graph, "eq", required_nodes, new_tasks)
                print("extra cost")
                print(str(iteration) + " __ " + str(extra_cost_1) + "_" + str(extra_cost_2))
                budget_it = budget_it + 1

            iteration = iteration + 1
            sh_previous_graph = shared_artifact_graph.copy()

    #graphviz_draw_with_requests(limited_shared_graph, "lt", limited_required_nodes)
    #graphviz_draw_with_requests(equivalent_graph, "eq", required_nodes)
# plot_artifact_graph(limited_shared_graph, uid, "limited")
# plot_artifact_graph(artifact_graph, uid, "shared")
# store_or_load_artifact_graph(equivalent_graph, sum, uid, 'eq_classifier', 'breast_cancer')
# store_EDGES_artifact_graph(limited_shared_graph, 100, uid, 'lt_classifier', 'breast_cancer', graph_dir="graphs")

# extract_nodes_and_edges()

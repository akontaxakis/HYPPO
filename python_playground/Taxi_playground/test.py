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

from Example.user_iterations import collab_HIGGS_all_operators, collab_TAXI_all_operators
from Dictionary.Outlier_removal.Taxi_DateTimeFeatures import CustomFeatureEngineer
from Dictionary.Outlier_removal.Taxi_OneHot import CustomOneHotEncoder
from Dictionary.Outlier_removal.Taxi_Outlier_Removal import Taxi_Outlier_Removal
from libs.parser import init_graph, add_load_tasks_to_the_graph, execute_pipeline, rank_based_materializer, \
    new_edges, extract_nodes_and_edges, \
    split_data, create_equivalent_graph, new_eq_edges, create_equivalent_graph_without_fit, graphviz_draw, \
    graphviz_draw_with_requests, graphviz_draw_with_requests_and_new_tasks
from libs.logical_pipeline_generator import logical_to_physical_random




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
    uid = "TAXI_ad_10_200_f6"
    iteration = 50
    k = 200
    N = 10



    categorical_features = ['store_and_fwd_flag', 'vendor_id']
    preprocessing_pipeline = Pipeline([
        ('OR', Taxi_Outlier_Removal()),
        ('OH', CustomOneHotEncoder(categorical_features)),
        ('FE', CustomFeatureEngineer()),
    ])
    pickle_path = f"{uid}_{iteration}_advance.gpickle"
    with open(pickle_path, 'rb') as file:
        shared_graph = pickle.load(file)

    # List of model file paths
    model_paths = [
        'taxi_models_f/X_SKCuSKCuSKGr2907',
        'taxi_models_f/X_SKCuSKCuSKKN0663',
        'taxi_models_f/X_SKCuSKCuSKRa2907',
        'taxi_models_f/X_SKTaSKCuSKCuSKStSKKN3874'
    ]

    # Define the number of models you want to select
    N = 2  # Replace with the number of models you want to select


    # Randomly select N paths
    selected_paths = random.sample(model_paths, N)

    # Load the selected models and store them in a list along with their names
    selected_models_with_names = [(path.split('/')[-1], joblib.load(path)) for path in selected_paths]

    # Now selected_models_with_names contains tuples of (model_name, model_object)
    for name, model in selected_models_with_names:
        print(f"Model Name: {name}, Model Object: {model}")


    X, y, raw_artifact_graph, cc = init_graph(dataset)
    X_test, X_train, y_test, y_train, cc = split_data(X, raw_artifact_graph, dataset, "no_sampling", y, cc)
    dataset_size = raw_artifact_graph.nodes[dataset]['size']

    preprocessed_data = preprocessing_pipeline.fit_transform(X_train)
    preprocessed_test = preprocessing_pipeline.transform(X_test)

    y_train_aligned = y_train[:len(preprocessed_data)]

    votingC = VotingRegressor(estimators=[selected_models_with_names],
                               n_jobs=4)

    votingC = votingC.fit(preprocessed_data, y_train_aligned)
    predictions = votingC.predict(preprocessed_test)


    #dataset = "breast_cancer"
    #uid = "TAXI_10_200_f5"

    operators_dict = {key: value for key, value in collab_TAXI_all_operators}
    #logical_pipelines_pool = "OR|OH|FE|;SS|RF();GBR();LGBM()|MSE"

    logical_pipelines_pool = ";OR|OH|FE|;SS|RF;LGBM;GBR;KNR;DTR|MSE;MAE"
    logical_pipelines_pool = "OR|OH|FE|SS|IM|RI;LGBM;GBR;KNR;LR|MSE;MAE;MPE"
    logical_pipelines_pool = "OR|OH|FE|SS|IM|RI;MS(2);MS(3);MS(4);MS(5)|VC;SM|MSE;MAE;MPE"


    dataset_size = shared_graph.nodes[dataset]['size']
    print(dataset_size)

    Budget = [0,dataset_size / 100000, dataset_size / 10000, dataset_size / 1000, dataset_size / 100, dataset_size, dataset_size * 10, dataset_size * 100]
    loading_speed = 566255240
    for i in range(N):

        Trails = logical_to_physical_random(logical_pipelines_pool,operators_dict, k)
        print(Trails)
        sh_previous_graph = shared_graph
        budget_it = 0
        iteration = 0
        for trial in Trails:
            #######################--SHARED GRAPH--#########################
            execution_graph, artifacts, request = execute_pipeline(dataset, raw_artifact_graph.copy(), uid, trial,
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
                print(str(extra_cost_1) + "_" + str(extra_cost_2))
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

import networkx as nx

from Components.augmenter import store_diff, new_edges, create_equivalent_graph_without_fit, new_eq_edges
from Components.history_manager import extract_nodes_and_edges, rank_based_materializer, add_load_tasks_to_the_graph
from Components.parser.parser import init_graph, split_data, execute_pipeline
from generator.general.logical_pipeline_generator import logical_to_physical_random

from python_playground.Example.user_iterations import collab_HIGGS_all_operators

if __name__ == '__main__':
    import os

    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    dataset = "HIGGS"
    # dataset = "breast_cancer"
    uid = "HIGGS"
    loading_speed = 566255240
    k = 50
    N = 5

    operators_dict = {key: value for key, value in collab_HIGGS_all_operators}
    logical_pipelines_pool = "SI|SS|;PF|SVM(1);SVM(0.5)|AC;F1;CA;KS"

    X, y, raw_artifact_graph, cc = init_graph(dataset)
    X_test, X_train, y_test, y_train, cc = split_data(X, raw_artifact_graph, dataset, "no_sampling", y, cc)
    dataset_size = raw_artifact_graph.nodes[dataset]['size']
    print(dataset_size)
    # Budget = [0, dataset_size/1000000, dataset_size/100000, dataset_size/10000, dataset_size/1000, dataset_size/100, dataset_size/10, dataset_size, dataset_size*10, dataset_size*100]
    # Budget = [0, dataset_size / 10, dataset_size, dataset_size * 10]
    # loading_speed = 5602400
    # Budget = [ dataset_size/1000]

    Budget = [0, dataset_size / 2000, dataset_size / 200, dataset_size / 20, dataset_size / 2]

    for i in range(N):

        Trails = logical_to_physical_random(logical_pipelines_pool, operators_dict, k)
        print(Trails)
        sh_previous_graph = raw_artifact_graph
        budget_it = 0
        iteration = 0
        for trial in Trails:
            #######################--SHARED GRAPH--#########################
            execution_graph, artifacts, request = execute_pipeline(dataset, raw_artifact_graph.copy(), uid, trial,
                                                                   "no_sampling", cc, X_train, y_train, X_test, y_test)
            shared_artifact_graph = nx.compose(execution_graph, sh_previous_graph)
            extract_nodes_and_edges(shared_artifact_graph, uid + "_" + str(i), "shared", iteration)

            ####################--Update Graphs for next iteration--#########
            budget_it = 0
            for b in Budget:
                sh_previous_graph_2 = sh_previous_graph.copy()

                ######################--LIMITED GRAPH--#########################
                limited_required_nodes, extra_cost_1, new_tasks = new_edges(sh_previous_graph, execution_graph)
                store_diff(limited_required_nodes, extra_cost_1, request, uid + "_" + str(budget_it) + "_" + str(i))
                materialized_artifacts_0 = rank_based_materializer(sh_previous_graph, b)
                limited_shared_graph = add_load_tasks_to_the_graph(sh_previous_graph, materialized_artifacts_0)
                extract_nodes_and_edges(limited_shared_graph, uid + "_" + str(budget_it) + "_" + str(i), "limited",
                                        iteration)
                # graphviz_draw_with_requests(limited_shared_graph, "lt", limited_required_nodes)
                # graphviz_draw_with_requests_and_new_tasks(limited_shared_graph, "lt", limited_required_nodes, new_tasks)
                ######################--EQUIVALENT GRAPH--######################

                # equivalent_graph, required_nodes, extra_cost = create_equivalent_graph_2(equivalent_graph,execution_graph.copy())

                equivalent_graph = create_equivalent_graph_without_fit(sh_previous_graph_2)
                required_artifacts, extra_cost_2, new_tasks = new_eq_edges(execution_graph, equivalent_graph, "no_fit")
                store_diff(required_artifacts, extra_cost_2, request, uid + "_eq_" + str(budget_it) + "_" + str(i))

                materialized_artifacts_1 = rank_based_materializer(equivalent_graph, b)
                equivalent_graph = add_load_tasks_to_the_graph(equivalent_graph, materialized_artifacts_1)
                extract_nodes_and_edges(equivalent_graph, uid + "_" + str(budget_it) + "_" + str(i), "equivalent",
                                        iteration)
                # graphviz_draw_with_requests(equivalent_graph, "eq", required_nodes)
                # graphviz_draw_with_requests_and_new_tasks(equivalent_graph, "eq", required_nodes, new_tasks)
                print("extra cost")
                print(str(iteration) + " __ " + str(extra_cost_1) + "_" + str(extra_cost_2))
                budget_it = budget_it + 1

            iteration = iteration + 1
            sh_previous_graph = shared_artifact_graph.copy()

    # graphviz_draw_with_requests(limited_shared_graph, "lt", limited_required_nodes)
    # graphviz_draw_with_requests(equivalent_graph, "eq", required_nodes)
# plot_artifact_graph(limited_shared_graph, uid, "limited")
# plot_artifact_graph(artifact_graph, uid, "shared")
# store_or_load_artifact_graph(equivalent_graph, sum, uid, 'eq_classifier', 'breast_cancer')
# store_EDGES_artifact_graph(limited_shared_graph, 100, uid, 'lt_classifier', 'breast_cancer', graph_dir="graphs")

# extract_nodes_and_edges()

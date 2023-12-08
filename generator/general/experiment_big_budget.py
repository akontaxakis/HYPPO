

from parser import init_graph, add_load_tasks_to_the_graph, plot_artifact_graph, \
    store_EDGES_artifact_graph, execute_pipeline, rank_based_materializer, new_edges, extract_nodes_and_edges, \
    split_data, create_equivalent_graph
from itertools import product
import random

from python_playground.Example.user_iterations import UR1_steps_3, UR1_steps_4, UR1_steps_1, UR2_steps_1, UR1_steps_2, \
    UR2_steps_2, all_steps


def store_diff(required_nodes, extra_cost, request, uid):
    os.makedirs('iterations_diff', exist_ok=True)
    with open('iterations_diff/' + uid + '_iterations_diff_' + str(iteration) + '.txt', 'w') as f:
        for node in required_nodes:
            f.write(str(node) + ",")
        f.write(str(extra_cost) + ",")
        f.write(str(request))
    print(required_nodes)

def generate_configurations(all_steps):
    step_names = [step[0].split(".")[1] if '.' in step[0] else step[0] for step in all_steps]
    step_options = [step[1] for step in all_steps]
    for values in product(*step_options):
        yield [(name, [option]) for name, option in zip(step_names, values)]  # generate tuples




if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")
    #Trails = [UR1_steps_0, UR1_steps_1,UR1_steps_2, UR2_steps_0, UR2_steps_1, UR2_steps_2]
    n = 100
    Trails = [UR1_steps_1, UR2_steps_1, UR1_steps_2, UR2_steps_2, UR1_steps_3, UR1_steps_4]
    #Trails = [UR2_steps_0, UR2_steps_1, UR2_steps_2]

    configurations = list(generate_configurations(all_steps))

    if len(configurations) >= n:
        Trails = random.sample(configurations, n)
    else:
        print("Not enough configurations to select 20. Total configurations:", len(configurations))
    Trails[0]=UR1_steps_3
    Trails[1]=UR1_steps_4


    #dataset = "commercial"
    dataset = "breast_cancer"
    uid = "ex100b"
    X, y, raw_artifact_graph, cc = init_graph(dataset)
    X_test, X_train, y_test, y_train,cc = split_data(X, raw_artifact_graph, dataset, "no_sampling", y, cc)


    dataset_size = raw_artifact_graph.nodes[dataset]['size']
    Budget = [0, dataset_size/1000000, dataset_size/100000, dataset_size/10000, dataset_size/1000, dataset_size/100, dataset_size/10, dataset_size]
    #Budget = [ dataset_size/1000]
    budget_it = 0
    for b in Budget:
        iteration = 0
        sh_previous_graph = raw_artifact_graph
        lt_previous_graph = raw_artifact_graph
        eq_previous_graph = raw_artifact_graph
        for trial in Trails:
                #######################--SHARED GRAPH--#########################
            shared_artifact_graph, artifacts, request = execute_pipeline(dataset, sh_previous_graph.copy(), uid, trial, "no_sampling", cc, X_train, y_train, X_test, y_test)
            extract_nodes_and_edges(shared_artifact_graph, uid+"_"+str(budget_it), "shared", iteration)
            extract_nodes_and_edges(shared_artifact_graph,uid+"_"+str(budget_it),"shared",iteration)

                 ######################--LIMITED GRAPH--#########################
            required_nodes, extra_cost = new_edges(sh_previous_graph, shared_artifact_graph)
            store_diff(required_nodes, extra_cost, request,uid+"_"+str(budget_it))
            materialized_artifacts_0 = rank_based_materializer(sh_previous_graph, b)
            limited_shared_graph = add_load_tasks_to_the_graph(sh_previous_graph, materialized_artifacts_0)
            extract_nodes_and_edges(limited_shared_graph, uid+"_"+str(budget_it),"limited", iteration)

                 ######################--EQUIVALENT GRAPH--######################
            equivalent_graph = create_equivalent_graph(sh_previous_graph)
            materialized_artifacts_1 = rank_based_materializer(equivalent_graph, b)
            equivalent_graph = add_load_tasks_to_the_graph(equivalent_graph, materialized_artifacts_1)
            extract_nodes_and_edges(equivalent_graph, uid+"_"+str(budget_it), "equivalent", iteration)

                ####################--Update Graphs for next iteration--#########

            iteration = iteration + 1
            sh_previous_graph = shared_artifact_graph.copy()
            lt_previous_graph = limited_shared_graph.copy()
        budget_it = budget_it + 1
    #plot_artifact_graph(limited_shared_graph, uid, "limited")
    #plot_artifact_graph(artifact_graph, uid, "shared")
    # store_or_load_artifact_graph(equivalent_graph, sum, uid, 'eq_classifier', 'breast_cancer')
    #store_EDGES_artifact_graph(limited_shared_graph, 100, uid, 'lt_classifier', 'breast_cancer', graph_dir="graphs")

   #extract_nodes_and_edges()

import pickle

import networkx as nx

from libs.artifact_graph_lib import load_artifact_graph, plot_artifact_graph, store_EDGES_artifact_graph

if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")

    G = nx.Graph()
    UD1= 'ef910ead'
    UD2 = '447856ae'
    artifact_graph_1 = load_artifact_graph(G, 43, UD1,'clustering','breast_cancer')
    artifact_graph_2 = load_artifact_graph(G, 177, UD2, 'classifier','breast_cancer')

    #G = nx.union(artifact_graph_1, artifact_graph_2,rename=('G1-', 'G2-'))
    G = nx.compose(artifact_graph_1, artifact_graph_2)
    plot_artifact_graph(G, "eq_"+UD1+"_"+UD2)
    sum = 43 + 177
    store_EDGES_artifact_graph(G, sum, "eq_"+UD1+"_"+UD2, 'clustering,classfier', 'breast_cancer')

    print("Number of nodes eq_graph:", G.number_of_nodes())
    print("Number of edges eq_graph:", G.number_of_edges())


    G = nx.Graph()
    UD1 = 'ef910ead'
    UD2 = '447856ae'
    artifact_graph_1 = load_artifact_graph(G, 43, UD1, 'clustering', 'breast_cancer',mode='')
    artifact_graph_2 = load_artifact_graph(G, 177, UD2, 'classifier', 'breast_cancer',mode='')

    G = nx.compose(artifact_graph_1, artifact_graph_2)
    plot_artifact_graph(G, UD1 + "_" + UD2)

    sum = 43 + 177
    store_EDGES_artifact_graph(G, sum, UD1 + "_" + UD2, 'clustering,classfier', 'breast_cancer')


    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
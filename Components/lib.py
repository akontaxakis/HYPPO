import os
import pickle
import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from Components.augmenter import map_node


def load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs", mode="eq_"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + mode + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    return artifact_graph


def store_EDGES_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs"):
    file_name = uid + "_EDGES_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.txt")
    with open(ag_path, "w") as outfile:
        # Iterate over edges and write to file
        for u, v, data in artifact_graph.edges(data=True):
            cost = data['weight']
            cost_2 = data['weight']
            # cost_3 = data['memory_usage']
            outfile.write(f'"{u}","{v}",{cost},{cost_2}\n')
            # outfile.write(f'"{u}","{v}",{cost},{cost_2},{cost_3}\n')
    return artifact_graph


def store_or_load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    else:
        with open(ag_path, 'wb') as f:
            pickle.dump(artifact_graph, f)

    file_name = uid + "_EDGES_AG_" + str(sum) + "_" + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.txt")

    with open(ag_path, "w") as outfile:
        # Iterate over edges and write to file
        for u, v, data in artifact_graph.edges(data=True):
            cost = data['cost']
            outfile.write(f'"{u}","{v}",{cost}\n')
    return artifact_graph


def create_artifact_graph(artifacts):
    G = nx.DiGraph()
    for i, (step_name, artifact) in enumerate(artifacts.items()):
        G.add_node(step_name, artifact=artifact)
        if i > 0:
            prev_step_name = list(artifacts.keys())[i - 1]
            G.add_edge(prev_step_name, step_name)
    return G


def plot_artifact_graph(G, uid, type):
    plt.figure(figsize=(20, 18))
    pos = nx.drawing.layout.spring_layout(G, seed=620, scale=4)
    nx.draw(G, pos, with_labels=True, node_size=120, node_color="skyblue", font_size=5)
    folder_path = "plots/"
    file_path = os.path.join(folder_path, uid + "_" + type + "_plot.pdf")
    plt.savefig(file_path)
    plt.show()


def load_artifact_graph(artifact_graph, sum, uid, objective, dataset, graph_dir="graphs", mode="eq_"):
    os.makedirs(graph_dir, exist_ok=True)
    file_name = uid + "_AG_" + str(sum) + "_" + mode + objective + "_" + dataset
    ag_path = os.path.join(graph_dir, f"{file_name}.pkl")
    if os.path.exists(ag_path):
        with open(ag_path, 'rb') as f:
            print("load " + ag_path)
            artifact_graph = pickle.load(f)
    return artifact_graph


def extract_artifact_graph(artifact_graph, graph_dir, uid):
    shared_graph_file = uid + "_shared_graph"
    shared_graph_path = os.path.join(graph_dir, f"{shared_graph_file}.plk")
    if os.path.exists(shared_graph_path):
        with open(shared_graph_path, 'rb') as f:
            print("load" + shared_graph_path)
            artifact_graph = pickle.load(f)
    return artifact_graph, shared_graph_path


def required_artifact(new_tasks):
    source_nodes = {edge[0] for edge in new_tasks}


def pretty_graph_drawing(G):
    graph_size = 5
    pos = nx.spring_layout(G)
    pos_1 = nx.spring_layout(G)
    depth = G.number_of_nodes();
    for node in G.nodes:
        G.nodes[node]['depth'] = None

    # Compute and set depth for each node
    compute_depth(G, 'source')

    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100

        elif G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['size'] = 10

        elif G.nodes[node_id]['type'] == 'super':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['size'] = 10


        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100

    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    print(pos)
    nx.draw(G, pos=pos, with_labels=True, font_size=2, node_shape="r", node_size=node_sizes, node_color=node_colors)
    plt.figure(figsize=(100, 100))

    plt.savefig("output.pdf", format="pdf")
    plt.show()


## graphviz styles can be found here: https://graphviz.org/docs/attr-types/style/
def graphviz_draw_with_requests_and_new_tasks(K, mode, requested_nodes, new_tasks):
    G = K.copy()

    for node in G.nodes:
        G.nodes[node]['depth'] = None

    # Compute and set depth for each node
    compute_depth(G, 'source')
    blue_nodes = []
    if mode == "eq":
        eq_requested_nodes = []
        for node in requested_nodes:
            eq_requested_nodes.append(map_node(node, "no_fit"))
        requested_nodes = eq_requested_nodes

    disconnected_nodes, disconnected_edges = find_disconnected_nodes_edges(G, requested_nodes)
    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        elif G.nodes[node_id]['type'] == 'super' or G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['edgecolors'] = 'blue'
            G.nodes[node_id]['shape'] = 'point'
            G.nodes[node_id]['width'] = 0.1
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'
        if node_id in requested_nodes:
            G.nodes[node_id]['color'] = 'black'
            G.nodes[node_id]['shape'] = 'ellipse'
        if node_id in disconnected_nodes:
            G.nodes[node_id]['style'] = "dotted"
        else:
            G.nodes[node_id]['style'] = "bold"

    labels = {node: "" if node in blue_nodes else str(node) for node in G.nodes()}
    for node, label in labels.items():
        G.nodes[node]['label'] = label

    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for edge in A.edges():
        edge.attr['label'] = int(G[edge[0]][edge[1]]['weight'] * 1000)
        if edge in disconnected_edges:
            edge.attr['style'] = "dotted"
        else:
            edge.attr['style'] = "bold"

    for u, v in new_tasks:
        if mode == "eq":
            A.add_edge(map_node(u, "no_fit"), map_node(v, "no_fit"))
        else:
            A.add_edge(u, v)

    # Legend
    # legend_label = "{ LEGEND | {Dotted Lines and Nodes |Pruned Elements} | {Black Ellipse|New Artifacts} | {Bold Black Ellipse|Requested Artifacts}}"
    #  A.add_node("Legend", shape="record", label=legend_label, rank='sink')

    # Ensure the legend is placed at the bottom
    # A.add_subgraph(["Legend"], rank="sink", name="cluster_legend")

    # Save the graph to a file
    file_path = "graph_output.png"
    A.layout(prog='dot')
    A.draw(file_path)

    # Open the saved image file with the default viewer
    if os.name == 'posix':
        os.system(f'open {file_path}')
    elif os.name == 'nt':  # For Windows
        os.startfile(file_path)

    # nx.draw(G, pos=pos, with_labels=True, font_size=2,node_shape="r", node_size=node_sizes, node_color=node_colors)
    # plt.figure(figsize=(100, 100))

    # plt.savefig("output.pdf", format="pdf")
    # plt.show()


def graphviz_draw_with_requests(G, mode, requested_nodes):
    for node in G.nodes:
        G.nodes[node]['depth'] = None

    # Compute and set depth for each node
    compute_depth(G, 'source')
    blue_nodes = []
    if mode == "eq":
        eq_requested_nodes = []
        for node in requested_nodes:
            eq_requested_nodes.append(map_node(node, "no_fit"))
        requested_nodes = eq_requested_nodes

    disconnected_nodes, disconnected_edges = find_disconnected_nodes_edges(G, requested_nodes)
    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        elif G.nodes[node_id]['type'] == 'super' or G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['edgecolors'] = 'blue'
            G.nodes[node_id]['shape'] = 'point'
            G.nodes[node_id]['width'] = 0.1
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'
        if node_id in requested_nodes:
            G.nodes[node_id]['color'] = 'black'
            G.nodes[node_id]['shape'] = 'ellipse'
        if node_id in disconnected_nodes:
            G.nodes[node_id]['style'] = "dotted"
        else:
            G.nodes[node_id]['style'] = "bold"

    labels = {node: "" if node in blue_nodes else str(node) for node in G.nodes()}
    for node, label in labels.items():
        G.nodes[node]['label'] = label
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for edge in A.edges():
        edge.attr['label'] = int(G[edge[0]][edge[1]]['weight'] * 10000)
        if edge in disconnected_edges:
            edge.attr['style'] = "dotted"
        else:
            edge.attr['style'] = "bold"
    # Save the graph to a file
    file_path = "graph_output.png"
    A.layout(prog='dot')
    A.draw(file_path)

    # Open the saved image file with the default viewer
    if os.name == 'posix':
        os.system(f'open {file_path}')
    elif os.name == 'nt':  # For Windows
        os.startfile(file_path)

    # nx.draw(G, pos=pos, with_labels=True, font_size=2,node_shape="r", node_size=node_sizes, node_color=node_colors)
    # plt.figure(figsize=(100, 100))

    # plt.savefig("output.pdf", format="pdf")
    # plt.show()


def graphviz_draw(G, mode):
    from IPython.display import Image

    # Convert the networkx graph to a pygraphviz graph

    # Customize appearance if needed
    # For example, you can modify node shapes, colors, edge types, etc.
    graph_size = 5
    pos = nx.spring_layout(G)
    pos_1 = nx.spring_layout(G)
    depth = G.number_of_nodes();
    for node in G.nodes:
        G.nodes[node]['depth'] = None

    # Compute and set depth for each node
    compute_depth(G, 'source')
    blue_nodes = []
    for node_id in G.nodes:
        if node_id == 'source':
            # pos[node_id] = np.array([0, depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), 0])
            G.nodes[node_id]['color'] = 'red'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        elif G.nodes[node_id]['type'] == 'split':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['size'] = 10
            G.nodes[node_id]['shape'] = 'rectangle'
            G.nodes[node_id]['shape'] = 'circle'
            G.nodes[node_id]['width'] = 0.3
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'super':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'blue'
            G.nodes[node_id]['shape'] = 'circle'
            G.nodes[node_id]['width'] = 0.3
            blue_nodes.append(node_id)

        elif G.nodes[node_id]['type'] == 'fitted_operator':
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'green'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

        else:
            # pos[node_id] = np.array([random.uniform(-2, 2), depth - G.nodes[node_id]['depth']])
            # pos[node_id] = np.array([-(depth - G.nodes[node_id]['depth']), random.uniform(-graph_size, graph_size)])
            G.nodes[node_id]['color'] = 'purple'
            G.nodes[node_id]['size'] = 100
            G.nodes[node_id]['shape'] = 'rectangle'

    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    labels = {node: "" if node in blue_nodes else str(node) for node in G.nodes()}
    for node, label in labels.items():
        G.nodes[node]['label'] = label
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr['rankdir'] = 'LR'
    for edge in A.edges():
        edge.attr['label'] = int(G[edge[0]][edge[1]]['weight'] * 10000)
    # Save the graph to a file
    file_path = "graph_output.png"
    A.layout(prog='dot')
    A.draw(file_path)

    # Open the saved image file with the default viewer
    if os.name == 'posix':
        os.system(f'open {file_path}')
    elif os.name == 'nt':  # For Windows
        os.startfile(file_path)

    # nx.draw(G, pos=pos, with_labels=True, font_size=2,node_shape="r", node_size=node_sizes, node_color=node_colors)
    # plt.figure(figsize=(100, 100))

    # plt.savefig("output.pdf", format="pdf")
    # plt.show()


def compute_depth(graph, node, parent=None):
    if parent is None:
        depth = 0
    else:
        depth = graph.nodes[parent]['depth'] + 1
    graph.nodes[node]['depth'] = depth
    for neighbor in graph.neighbors(node):
        if neighbor != parent:
            compute_depth(graph, neighbor, node)


def find_disconnected_nodes_edges(G, targets):
    connected_nodes = set()
    for node in G.nodes():
        for target in targets:
            if nx.has_path(G, node, target):
                connected_nodes.add(node)
                break

    # Step 2: Identify edges connected to these nodes
    connected_edges = {(u, v) for u, v in G.edges() if v in connected_nodes}

    # Step 3: Return nodes and edges that aren't in the above sets
    disconnected_nodes = set(G.nodes()) - connected_nodes
    disconnected_edges = set(G.edges()) - connected_edges
    return disconnected_nodes, disconnected_edges


def get_first_lines(filename, n=10):
    """
    Extract the first n lines of a file.

    Parameters:
    - filename: path to the file
    - n: number of lines to extract

    Returns:
    - list of the first n lines
    """

    with open(filename, 'r', encoding="utf-8") as f:
        lines = [next(f) for _ in range(n)]

    return lines

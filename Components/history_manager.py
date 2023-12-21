import heapq
import os


def rank_based_materializer(artifact_graph, Budget):
    pq = []
    materialized_artifacts = []
    size_sofar = 0
    if Budget == 0:
        return materialized_artifacts
    for node_id, attrs in artifact_graph.nodes(data=True):
        if attrs['type'] not in ["super", "raw", "split", 'source']:
            priority = -1 * attrs['cc'] / attrs['size']
            heapq.heappush(pq, (priority, node_id))
    while pq:
        priority, item = heapq.heappop(pq)
        if Budget > size_sofar + artifact_graph.nodes[item]['size']:
            materialized_artifacts.append(item)
            size_sofar = size_sofar + artifact_graph.nodes[item]['size']

    return materialized_artifacts


def rank_based_materializer_frequency(artifacts, Budget):
    pq = []
    materialized_artifacts = []
    size_sofar = 0
    if Budget == 0:
        return materialized_artifacts
    for node_id, attrs in artifacts(data=True):
        if attrs['type'] not in ["super", "raw", "split", 'source']:
            priority = -1 * (attrs['frequency'] * attrs['cc']) / attrs['size']
            heapq.heappush(pq, (priority, node_id))
    while pq:
        priority, item = heapq.heappop(pq)
        if Budget > size_sofar + artifacts[item]['size']:
            materialized_artifacts.append(item)
            size_sofar = size_sofar + artifacts[item]['size']

    return materialized_artifacts


def add_load_tasks_to_the_graph(shared_artifact_graph, materialized_artifacts, loading_speed):
    platforms = []
    platforms.append("python")
    limited_shared_graph = shared_artifact_graph.copy()
    for artifact in materialized_artifacts:
        loading_time = limited_shared_graph.nodes[artifact]['size'] / loading_speed
        limited_shared_graph.add_edge("source", artifact, type='load', weight=loading_time, execution_time=loading_time,
                                      memory_usage=0, platform=platforms)
    return limited_shared_graph


def update_and_merge_graphs(H, E):
    """
    Updates the 'frequency' attribute for nodes in H that also exist in E,
    and then adds any nodes and edges from E that don't exist in H.

    Parameters:
    H (nx.DiGraph): The first directed graph, to be updated.
    E (nx.DiGraph): The second directed graph, from which data is added to H.

    Returns:
    nx.DiGraph: The updated graph H.
    """
    # Update frequency attribute for nodes that exist in both H and E
    for node in H.nodes():
        if node in E:
            # Increment frequency attribute if it exists, otherwise initialize it
            H.nodes[node]['frequency'] = H.nodes[node].get('frequency', 0) + 1

    # Add nodes and edges from E that don't exist in H
    for node in E.nodes():
        if node not in H:
            H.add_node(node, **E.nodes[node])

    for edge in E.edges(data=True):
        if not H.has_edge(*edge[:2]):
            H.add_edge(*edge[:2], **edge[2])

    return H


def extract_nodes_and_edges(artifact_graph, uid, type, iteration, graph_dir='graphs/iteration_graphs'):
    os.makedirs(graph_dir, exist_ok=True)
    graph_file = uid + "_" + type + "_" + str(iteration)
    shared_graph_path = os.path.join(graph_dir, graph_file)
    os.makedirs(shared_graph_path, exist_ok=True)
    if os.path.exists(shared_graph_path):
        with open(shared_graph_path + '/nodes.txt', 'w') as f:
            for node in artifact_graph.nodes(data=True):
                f.write(str(node) + "\n")
        with open(shared_graph_path + '/edges.txt', 'w') as f:
            for node in artifact_graph.edges(data=True):
                f.write(str(node) + "\n")

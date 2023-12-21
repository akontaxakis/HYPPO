import os


def change_node_name(G, old_name, new_name):
    # Copy the attributes of the old node to the new node
    attributes = G.nodes[old_name]
    G.add_node(new_name, **attributes)

    # Iterate through the edges of the old node and add equivalent edges with the new node
    for neighbor, edge_attrs in G[old_name].items():
        G.add_edge(new_name, neighbor, **edge_attrs)

    # Remove the old node from the graph
    G.remove_node(old_name)


def store_diff(required_nodes, extra_cost, request, uid, iteration):
    os.makedirs('iterations_diff', exist_ok=True)
    with open('iterations_diff/' + uid + '_iterations_diff_' + str(iteration) + '.txt', 'w') as f:
        for node in required_nodes:
            f.write(str(node) + ",")
        f.write(str(extra_cost) + ",")
        f.write(str(request))
    print(required_nodes)


def create_equivalent_graph_without_fit(artifact_graph_2):
    artifact_graph = artifact_graph_2.copy()
    artifact_graph = merge_EQ_nodes_without_fit(artifact_graph)
    return artifact_graph


# used
def new_eq_edges(execution_graph, equivalent_graph, mode):
    # Get the edges from both graphs
    new_tasks = []
    additional_cost = 0
    produced_artifacts = []
    eq_edges = equivalent_graph.edges
    for u, v in execution_graph.edges():
        u_m = map_node(u, mode)
        v_m = map_node(v, mode)
        if (u_m, v_m) in equivalent_graph.edges():
            ex_platform = execution_graph[u][v]['platform']
            eq_platforms = equivalent_graph[u_m][v_m]['platform']
            if ex_platform[0] not in eq_platforms:
                additional_cost = additional_cost + execution_graph[u][v]['weight']
                produced_artifacts.append(v)
                new_tasks.append((u, v))
        else:
            additional_cost = additional_cost + execution_graph[u][v]['weight']
            new_tasks.append((u, v))
            produced_artifacts.append(v)

    source_nodes = {edge[0] for edge in new_tasks}
    final_set = []
    for node in source_nodes:
        if execution_graph.nodes[node]['type'] != 'super':
            if equivalent_graph.has_node(map_node(node, mode)):
                if node not in produced_artifacts:
                    final_set.append(node)
    if len(final_set) == 0:
        nodes_without_outgoing_edges = [node for node, outdegree in execution_graph.out_degree() if outdegree == 0]
        for node in nodes_without_outgoing_edges:
            if equivalent_graph.has_node(map_node(node, mode)):
                final_set.append(node)
    # Find the difference
    return final_set, additional_cost, new_tasks


# used
def new_edges(artifact_graph_0, artifact_graph_1):
    # Get the edges from both graphs
    tasks_0 = set(artifact_graph_0.edges())
    tasks_1 = set(artifact_graph_1.edges())
    diff = tasks_1 - tasks_0
    extra_cost = 0
    for edge in diff:
        extra_cost = extra_cost + artifact_graph_1.edges[edge]['weight']
    source_nodes = {edge[0] for edge in diff}
    final_set = []
    for node in source_nodes:
        if artifact_graph_1.nodes[node]['type'] != 'super':
            if artifact_graph_0.has_node(node):
                final_set.append(node)
    # Find the difference
    if len(final_set) == 0:
        nodes_without_outgoing_edges = [node for node, outdegree in artifact_graph_1.out_degree() if outdegree == 0]
        for node in nodes_without_outgoing_edges:
            if artifact_graph_0.has_node(node):
                final_set.append(node)
    return final_set, extra_cost, diff


import re


def remove_prefixes(s):
    # Find all the occurrences of the prefixes
    matches = list(re.finditer(r'GP|TF|TR|SK|GL', s))

    # If there are matches, remove them all from the string
    if matches:
        # Since we'll be adjusting the string and altering its length,
        # we need to compute the position offsets while removing matches
        offset = 0
        for match in matches[:-1]:
            start = match.start() - offset
            end = match.end() - offset
            s = s[:start] + s[end:]
            offset += (end - start)
    return s


def map_node(node, mode):
    if mode == "no_fit" and (("_fit_" in node) or ("_fit" in node)):
        modified_node = node
        modified_node = remove_prefixes(modified_node)
    else:
        modified_node = node.replace("GP", "")
        modified_node = modified_node.replace("TF", "")
        modified_node = modified_node.replace("GL", "")
        modified_node = modified_node.replace("TR", "")
        modified_node = modified_node.replace("SK", "")
    return modified_node


def create_equivalent_graph(artifact_graph_2):
    artifact_graph = artifact_graph_2.copy()
    artifact_graph = merge_EQ_nodes(artifact_graph)
    return artifact_graph


def merge_EQ_nodes_without_fit(artifact_graph):
    nodes_to_remove = []
    nodes = artifact_graph.nodes()
    modified_graph = artifact_graph.copy()
    for node in nodes:
        # if artifact_graph.nodes[node]['type'] != "super":
        if ("_fit_" not in node) and ("_fit" not in node) and (
                "GL" in node or "GP" in node or "TF" in node or "TR" in node or "SK" in node):
            modified_node = node.replace("GP", "")
            modified_node = modified_node.replace("TF", "")
            modified_node = modified_node.replace("TR", "")
            modified_node = modified_node.replace("SK", "")
            modified_node = modified_node.replace("GL", "")
            if not modified_graph.has_node(modified_node):
                modified_graph.add_node(modified_node, **artifact_graph.nodes[node])
            s1 = modified_graph.nodes[node]['size']
            s2 = modified_graph.nodes[modified_node]['size']
            f1 = modified_graph.nodes[node]['frequency']
            f2 = modified_graph.nodes[modified_node]['frequency']
            modified_graph = merge_nodes_3(modified_graph, modified_node, node)
            modified_graph.nodes[modified_node]['size'] = min(s1, s2)
            modified_graph.nodes[modified_node]['frequency'] = sum([f1, f2])

    modified_graph_2 = modified_graph.copy()
    nodes2 = modified_graph.nodes()
    for node2 in nodes2:
        # print(node2)
        if "SK" in node2 or "TF" in node2 or "GL" in node2 or "TR" in node2 or "GP" in node2 and "_fit" in node2 or "_fit_" in node2:
            tmp_node = remove_prefixes(node2)
            if not modified_graph_2.has_node(tmp_node):
                modified_graph_2.add_node(tmp_node, **modified_graph.nodes[node2])
            s1 = modified_graph_2.nodes[node2]['size']
            s2 = modified_graph_2.nodes[tmp_node]['size']
            if tmp_node != node2:
                modified_graph_2 = merge_nodes_3(modified_graph_2, tmp_node, node2)
            # modified_graph.nodes[tmp_node]['size'] = min(s1, s2)
    return modified_graph_2


def merge_EQ_nodes(artifact_graph):
    nodes_to_remove = []
    nodes = artifact_graph.nodes()
    modified_graph = artifact_graph.copy()
    for node in nodes:
        # if artifact_graph.nodes[node]['type'] != "super":
        if "GP" in node or "TF" in node or "TR" in node or "GL" in node:
            modified_node = node.replace("GP", "")
            modified_node = modified_node.replace("TF", "")
            modified_node = modified_node.replace("GL", "")
            modified_node = modified_node.replace("TR", "")

            if modified_node in nodes:
                # Create new node with modified label (without "GPU")
                # if modified_node in artifact_graph.nodes():
                # print("current")
                # print(artifact_graph.in_edges(node, data=True))
                s1 = artifact_graph.nodes[node]['size']
                # print(artifact_graph.in_edges(modified_node, data=True))
                s2 = artifact_graph.nodes[modified_node]['size']
                # artifact_graph = nx.contracted_nodes(artifact_graph, modified_node, node)
                modified_graph = merge_nodes_3(modified_graph, modified_node, node)
                # print("modified")
                print(modified_graph.in_edges(modified_node, data=True))
                print(modified_graph.out_edges(modified_node, data=True))
                modified_graph.nodes[modified_node]['size'] = min(s1, s2)
    return modified_graph


def merge_nodes(G: object, node1: object, node2: object, nodes_to_remove) -> object:
    # Combine the neighbors of both nodes
    out_neighbors = set(G.successors(node1)).union(G.successors(node2))
    in_neighbors = set(G.predecessors(node1)).union(G.predecessors(node2))

    # Iterate through the out-neighbors, adding edges between node1 and the neighbors
    for neighbor in out_neighbors:
        if neighbor == node1 or neighbor == node2:
            continue

        if G.has_edge(node1, neighbor) and G.has_edge(node2, neighbor):
            weight = min(G[node1][neighbor]['weight'], G[node2][neighbor]['weight'])
        elif G.has_edge(node1, neighbor):
            weight = G[node1][neighbor]['weight']
        else:
            weight = G[node2][neighbor]['weight']

        G.add_edge(node1, neighbor, weight=weight)
    # Iterate through the in-neighbors, adding edges between the neighbors and node1
    for neighbor in in_neighbors:
        if neighbor == node1 or neighbor == node2:
            continue

        if G.has_edge(neighbor, node1) and G.has_edge(neighbor, node2):
            weight = min(G[neighbor][node1]['weight'], G[neighbor][node2]['weight'])
        elif G.has_edge(neighbor, node1):
            weight = G[neighbor][node1]['weight']
        else:
            weight = G[neighbor][node2]['weight']

        G.add_edge(neighbor, node1, weight=weight)

    # Remove node2 from the graph
    nodes_to_remove.append(node2)
    # G.remove_node(node2)
    return G, nodes_to_remove


def merge_nodes_3(G: object, node1: object, node2: object) -> object:
    # Combine the neighbors of both nodes
    out_neighbors = set(G.successors(node1)).union(G.successors(node2))
    in_neighbors = set(G.predecessors(node1)).union(G.predecessors(node2))

    # Add edges between node1 and the neighbors
    for neighbor in out_neighbors:
        if neighbor in {node1, node2}: continue

        weight = min(G[node1].get(neighbor, {}).get('weight', float('inf')),
                     G[node2].get(neighbor, {}).get('weight', float('inf')))
        mem = min(G[node1].get(neighbor, {}).get('memory_usage', 0),
                  G[node2].get(neighbor, {}).get('memory_usage', 0))
        list1 = G[node1].get(neighbor, {}).get('platform', [])
        list2 = G[node2].get(neighbor, {}).get('platform', [])
        combined = list(set(list1 + list2))
        edge_type = G[node1].get(neighbor, {}).get('type', 'super')
        if G.has_edge(node1, neighbor):
            # Update edge if it already exists
            G[node1][neighbor]['weight'] = weight
            G[node1][neighbor]['execution_time'] = weight
            G[node1][neighbor]['platform'] = combined
            G[node1][neighbor]['type'] = edge_type
            G[node1][neighbor]['memory_usage'] = mem
        else:
            # Add new edge otherwise
            G.add_edge(node1, neighbor, type=edge_type, weight=weight, execution_time=weight, memory_usage=mem,
                       platform=combined)

    # Add edges between the neighbors and node1
    for neighbor in in_neighbors:
        if neighbor in {node1, node2}: continue

        weight = min(G[neighbor].get(node1, {}).get('weight', float('inf')),
                     G[neighbor].get(node2, {}).get('weight', float('inf')))
        mem = min(G[neighbor].get(node1, {}).get('memory_usage', 0),
                  G[neighbor].get(node2, {}).get('memory_usage', 0))
        list1 = G[neighbor].get(node1, {}).get('platform', [])
        list2 = G[neighbor].get(node2, {}).get('platform', [])
        combined = list(set(list1 + list2))
        edge_type = G[neighbor].get(node1, {}).get('type', 'super')
        if G.has_edge(neighbor, node1):
            # Update edge if it already exists
            G[neighbor][node1]['weight'] = weight
            G[neighbor][node1]['execution_time'] = weight
            G[neighbor][node1]['platform'] = combined
            G[neighbor][node1]['type'] = edge_type
            G[neighbor][node1]['memory_usage'] = mem
        else:
            # Add new edge otherwise
            G.add_edge(neighbor, node1, type=edge_type, weight=weight, execution_time=weight, memory_usage=mem,
                       platform=combined)

    # Remove node2 from the graph
    G.remove_node(node2)

    return G

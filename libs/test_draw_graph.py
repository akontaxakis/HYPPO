import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import re


def remove_prefixes(s):
    # Find all the occurrences of the prefixes
    matches = list(re.finditer(r'GP|TF|TR', s))

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
if __name__ == '__main__':
    import networkx as nx
    import matplotlib.pyplot as plt

    # Create a graph
    G = nx.DiGraph()

    # Add nodes
    G.add_node("source",pos=1, size=1000, color='red')  # Node 1 will be smaller and red
    G.add_node("kappa",pos=2, size=300, color='blue')  # Node 2 will be larger and blue
    G.add_node("lamda",pos=3, size=200, color='green')  # Node 3 will be smaller and green
    pos = nx.spring_layout(G)
    print(pos)
    pos['source'] = np.array([0, 10])
    print(pos)
    # Add edges
    G.add_edge("source","kappa")
    G.add_edge("source", "lamda")

    # Set the "source" node that you want on top
    source_node = "source"

    # Extract node attributes
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]

    # Specify the order of node drawing, putting the "source" node on top

    #nx.draw_networkx(G, pos=pos)
    # Draw the graph with customized node sizes and colors, respecting the drawing order
    # You can choose a layout algorithm that suits your graph
    nx.draw(G, pos=pos, with_labels=True, node_size=node_sizes, node_color=node_colors,
            font_weight='bold')

    # Display the graph
    plt.show()

    import networkx as nx
    import matplotlib.pyplot as plt

    # Create a sample graph
    G = nx.Graph()
    G.add_node(1)

    # Draw the graph with square nodes
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color='blue', node_size=300, node_shape='s')
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.axis('off')
    plt.show()




    input_str = "X_TFSiTRStGPPCTRML_fit_Psuper"
    result = remove_prefixes(input_str)
    print(result)

    G = nx.DiGraph()
    G.add_node(1)
    for node in G.nodes():
        G.add_node(2)




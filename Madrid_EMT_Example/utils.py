import pandas as pd
from pymnet import MultilayerNetwork,draw
import matplotlib.pyplot as plt

NODES_PATH = '../Data/Final_data/madrid_transport_nodes.csv'
SHP_NODES_PATH = '../Data/Final_data/final_nodes.csv'
EDGES_PATH = '../Data/Final_data/madrid_transport_edges.csv'
EDGES_NO_WEIGHT_PATH = '../Data/Final_data/madrid_transport_edges_no_weight.csv'
MADRID_TRANSPORT_NODES_SHP = pd.read_csv(SHP_NODES_PATH)


MADRID_NODE_COORDS = {
    row['stop']: (row['stop_lon'], row['stop_lat'])  # X = lon, Y = lat
    for _, row in MADRID_TRANSPORT_NODES_SHP.iterrows()
}




def read_multiplex_node_layer_edges_from_csv(file_path, weighted=False, directed=False):
    """
    Reads multilayer edges from a CSV in node,layer,node,layer format and returns a pymnet network.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    weighted : bool
        If True, assumes the last column is edge weight.
    directed : bool
        If True, creates a directed network.
    couplings : str
        Coupling type for pymnet network (only applies to Multiplex).

    Returns
    -------
    MultilayerNetwork
    """
    # Load data
    df = pd.read_csv(file_path)

    # Handle weights if present
    if weighted:
        weights = df.iloc[:, -1]
        edge_data = df.iloc[:, :-1]
    else:
        weights = pd.Series([1] * len(df))
        edge_data = df

    if edge_data.shape[1] != 4:
        raise ValueError("Expected columns: node1, layer1, node2, layer2 (plus optional weight).")

    # Initialize general multilayer network
    net = MultilayerNetwork(directed=directed, aspects=1)

    # Add edges
    for (_, row), weight in zip(edge_data.iterrows(), weights):
        u = (row[0], row[1])  # (node1, layer1)
        v = (row[2], row[3])  # (node2, layer2)
        net[u][v] = weight

    return net

def read_multiplex_edges_from_csv(file_path, weighted=False, directed=False):
    """
    Reads multilayer edges from a CSV and returns a pymnet network.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    weighted : bool
        If True, assumes the last column is edge weight.
    directed : bool
        If True, creates a directed network.

    Returns
    -------
    MultiplexNetwork
    """
    # Load data
    df = pd.read_csv(file_path)

    # Handle weights if present
    if weighted:
        weights = df.iloc[:, -1]
        edge_data = df.iloc[:, :-1]
    else:
        weights = pd.Series([1] * len(df))
        edge_data = df.iloc[:, :]
    # Detect number of layers from column count
    n_cols = edge_data.shape[1]
    if n_cols % 2 != 0:
        raise ValueError("Number of columns must be even (pairs of node-layer).")

    # Initialize network
    net = MultilayerNetwork(directed=directed, aspects=1)
    #min wight > 0
    min_weight = weights[weights > 0].min() if weighted else 1


    # Add edges
    for (_, row), weight in zip(edge_data.iterrows(), weights):
        # net[tuple(row)] =1 

        net[tuple(row)] = min_weight/weight if weight > 0 else 0

    return net



def plot_selected_multilayer_map(mnet, node_coords, important_nodes, output_path='madrid_transport_network.png'):
    """
    Plot a multilayer transportation network highlighting selected important nodes.

    Args:
        mnet (MultilayerNetwork): The multilayer network object (e.g., from pymnet).
        node_coords (dict): Coordinates for plotting nodes, e.g. from shapefile.
        important_nodes (list of str): Nodes to highlight with custom color and size.
        output_path (str): Path to save the output image (default is 'madrid_transport_network.png').
    """
    # Custom styling for important nodes
    custom_node_labels = {node: node for node in important_nodes}
    custom_node_colors = {node: 'red' for node in important_nodes}

    # Visual parameters
    node_alpha = {
        (node, layer): 0.6 if node in custom_node_labels else 0.4
        for (node, layer) in mnet.iter_node_layers()
    }
    node_size = {
        (node, layer): 0.01 if node in custom_node_labels else 0.005
        for (node, layer) in mnet.iter_node_layers()
    }
    node_label_dict = {
        (node, layer): custom_node_labels[node]
        for (node, layer) in mnet.iter_node_layers()
        if node in custom_node_labels
    }
    node_color_dict = {
        (node, layer): custom_node_colors[node]
        for (node, layer) in mnet.iter_node_layers()
        if node in custom_node_colors
    }

    # Plotting
    fig = plt.figure(figsize=(36, 24))
    ax_n = fig.add_subplot(111, projection='3d')
    draw(
        mnet,
        nodeLabelDict=node_label_dict,
        nodeLabelRule={},
        nodeCoords=node_coords,
        defaultNodeLabel="",
        defaultLayerLabelLoc=(-0.1, 0.1),
        defaultEdgeAlpha=0.4,
        defaultNodeLabelAlpha=1,
        defaultNodeLabelColor='red',
        nodeColorDict=node_color_dict,
        nodeSizeDict=node_size,
        layergap=1,
        ax=ax_n
    )

    fig.savefig(output_path, dpi=300, bbox_inches='tight')



from pymnet import draw
from Multinets.community.leiden import leiden_algorithm, generate_leiden_communities_net
import matplotlib.pyplot as plt
from Madrid_EMT_Example.utils import read_multiplex_edges_from_csv,EDGES_PATH,MADRID_NODE_COORDS


def plot_leiden_community_visualizations(
    mnet_weighted,
    nodes_shp,
    gamma=0.0005,
    output_prefix='leiden',
    scale_factor=1300,
    node_size_global=0.01
):
    """
    Run the Leiden community detection algorithm and plot both:
    1. A network of communities scaled by size.
    2. The original network with community membership highlighted.

    Args:
        mnet_weighted (MultilayerNetwork): Weighted multilayer network.
        nodes_shp (GeoDataFrame): DataFrame with 'stop', 'stop_lon', 'stop_lat' columns.
        gamma (float): Resolution parameter for the Leiden algorithm.
        output_prefix (str): Prefix for output image filenames.
        scale_factor (float): Scaling divisor for node sizes based on community size.
        node_size_global (float): Uniform node size for the original network.

    Saves:
        - '<output_prefix>_communities_scaled_gamma_<gamma>.png'
        - '<output_prefix>_highlighted_nodes_gamma_<gamma>.png'
    """
    # Run Leiden algorithm
    leiden_states = leiden_algorithm(mnet_weighted, gamma=gamma, max_iterations=100)
    print(f"Leiden iterations: {len(leiden_states)}")
    print("Communities found:", len(leiden_states[-1]['com2nodes']))

    # Generate community network
    leiden_com_net, leiden_com_sizes, leiden_com_2_name = generate_leiden_communities_net(
        mnet_weighted, leiden_states[-1]
    )

    # Node sizing and coloring for community network
    nodeSizeDict_leiden = {
        (node, layer): leiden_com_sizes[node] / scale_factor
        for (node, layer) in leiden_com_net.iter_node_layers()
        if node in leiden_com_sizes
    }

    colors = plt.get_cmap('tab20', len(leiden_com_net.slices[0]))
    nodeColorDict_leiden = {
        (node, layer): colors(i)
        for i, node in enumerate(leiden_com_net.slices[0])
        for (node_, layer) in leiden_com_net.iter_node_layers()
        if node_ == node
    }

    # Generate coordinates from shapefile
    node_coords = {
        row['stop']: (row['stop_lon'], row['stop_lat'])
        for _, row in nodes_shp.iterrows()
    }

    # Community color mapping for original network
    nodeColorDict_global = {
        node: nodeColorDict_leiden[(leiden_com_2_name[com], 1)]
        for com, nodes in leiden_states[-1]['com2nodes'].items()
        for node in nodes
    }

    nodeSizeDict_global = {
        (node, layer): node_size_global
        for (node, layer) in mnet_weighted.iter_node_layers()
    }

    # --- Plot 1: Community network ---
    fig1 = plt.figure(figsize=(36, 24))
    ax1 = fig1.add_subplot(111, projection='3d')

    draw(
        leiden_com_net,
        nodeCoords=node_coords,
        nodeColorDict=nodeColorDict_leiden,
        defaultNodeLabel="",
        defaultEdgeAlpha=0.4,
        nodeSizeDict=nodeSizeDict_leiden,
        layergap=1,
        ax=ax1
    )

    fig1.savefig(f'{output_prefix}_communities_scaled_gamma_{gamma}.png', dpi=300, bbox_inches='tight')

    # --- Plot 2: Original network with colored nodes ---
    fig2 = plt.figure(figsize=(36, 24))
    ax2 = fig2.add_subplot(111, projection='3d')

    draw(
        mnet_weighted,
        nodeCoords=node_coords,
        nodeSizeDict=nodeSizeDict_global,
        nodeColorDict=nodeColorDict_global,
        nodeLabelRule={},
        defaultNodeLabel="",
        defaultEdgeAlpha=0.3,
        layergap=1,
        ax=ax2
    )

    fig2.savefig(f'{output_prefix}_highlighted_nodes_gamma_{gamma}.png', dpi=300, bbox_inches='tight')


mnet_weighted = read_multiplex_edges_from_csv(EDGES_PATH, weighted=True)

plot_leiden_community_visualizations(
    mnet_weighted,
    MADRID_NODE_COORDS,
    gamma=0.0005,
    output_prefix='leiden',
    scale_factor=1300,
    node_size_global=0.01
)

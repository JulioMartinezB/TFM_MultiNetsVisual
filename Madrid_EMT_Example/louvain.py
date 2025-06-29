from pymnet import draw
from Multinets.community.louvain import louvain_algorithm, generate_louvain_communities_net
import matplotlib.pyplot as plt
from Madrid_EMT_Example.utils import read_multiplex_edges_from_csv,EDGES_PATH,MADRID_NODE_COORDS

def plot_louvain_community_visualizations(
    mnet,
    nodes_shp,
    output_prefix='imgs/louvain',
    scale_factor=1000,
    min_label_size=10,
    node_size_global=0.01
):
    """
    Run Louvain community detection and generate visualizations:
    1. Scaled community network.
    2. Original network with communities colored.

    Args:
        mnet (MultilayerNetwork): The unweighted multilayer network.
        nodes_shp (pd.DataFrame): GeoDataFrame with 'stop', 'stop_lon', 'stop_lat' columns.
        output_prefix (str): Base path for output images.
        scale_factor (float): Divisor for community node size scaling.
        min_label_size (int): Minimum community size to show label.
        node_size_global (float): Uniform node size for original network plot.

    Saves:
        - <output_prefix>_communities_scaled.png
        - <output_prefix>_highlighted_nodes.png
    """
    # Run Louvain algorithm
    louvain_states = louvain_algorithm(mnet)
    print(f"Louvain iterations: {len(louvain_states)}")
    print(f"Communities found: {len(louvain_states[-1]['com2nodes'])}")

    # Build community network
    louvain_com_net, louvain_com_sizes, louvain_com_2_name = generate_louvain_communities_net(
        mnet, louvain_states[-1]
    )

    # Node size dicts
    nodeSizeDict_louvain = {
        (node, layer): louvain_com_sizes[node] / scale_factor
        for (node, layer) in louvain_com_net.iter_node_layers()
        if node in louvain_com_sizes
    }
    nodeSizeDict_global = {
        (node, layer): node_size_global
        for (node, layer) in mnet.iter_node_layers()
    }

    # Coordinates from shapefile
    node_coords = {
        row['stop']: (row['stop_lon'], row['stop_lat'])
        for _, row in nodes_shp.iterrows()
    }

    # Community colors
    colors = plt.get_cmap('tab20', len(louvain_com_net.slices[0]))
    community_color_map = {
        node: colors(i)
        for i, node in enumerate(louvain_com_net.slices[0])
        if node in louvain_com_sizes
    }
    nodeColorDict_louvain = {
        (node, layer): community_color_map[node]
        for (node, layer) in louvain_com_net.iter_node_layers()
        if node in community_color_map
    }

    # Node colors in original network based on community color
    nodeColorDict_global = {
        node: nodeColorDict_louvain[(louvain_com_2_name[com], 1)]
        for com, nodes in louvain_states[-1]['com2nodes'].items()
        for node in nodes
    }

    # Label only large communities
    nodeLabelDict_louvain = {
        (node, layer): node
        for (node, layer) in louvain_com_net.iter_node_layers()
        if node in louvain_com_sizes and louvain_com_sizes[node] >= min_label_size
    }

    # --- Plot 1: Community Network ---
    fig1 = plt.figure(figsize=(36, 24))
    ax1 = fig1.add_subplot(111, projection='3d')
    draw(
        louvain_com_net,
        nodeCoords=node_coords,
        nodeColorDict=nodeColorDict_louvain,
        nodeLabelDict=nodeLabelDict_louvain,
        nodeLabelRule={},
        defaultNodeLabel="",
        defaultEdgeAlpha=0.4,
        nodeSizeDict=nodeSizeDict_louvain,
        layergap=1,
        ax=ax1
    )
    fig1.savefig(f'{output_prefix}_communities_scaled.png', dpi=300, bbox_inches='tight')

    # --- Plot 2: Original Network with Community Coloring ---
    fig2 = plt.figure(figsize=(36, 24))
    ax2 = fig2.add_subplot(111, projection='3d')
    draw(
        mnet,
        nodeCoords=node_coords,
        nodeColorDict=nodeColorDict_global,
        nodeSizeDict=nodeSizeDict_global,
        nodeLabelRule={},
        defaultNodeLabel="",
        defaultEdgeAlpha=0.3,
        layergap=1,
        ax=ax2
    )
    fig2.savefig(f'{output_prefix}_highlighted_nodes.png', dpi=300, bbox_inches='tight')

mnet_weighted = read_multiplex_edges_from_csv(EDGES_PATH, MADRID_NODE_COORDS)

plot_louvain_community_visualizations(
    mnet_weighted,
    MADRID_NODE_COORDS,
    output_prefix='imgs/louvain',
    scale_factor=1300,
    node_size_global=0.01
)
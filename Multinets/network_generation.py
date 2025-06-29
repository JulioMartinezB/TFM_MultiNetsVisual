import random
from pymnet import MultilayerNetwork, MultiplexNetwork


def er_general_multilayer(
    n_layers_per_aspect : list,
    n_nodes : int,
    p=0.5,
    randomWeights=False,
    directed=False
):
    """
    Generate an Erdős-Rényi-style random multilayer network.

    Each layer combination across aspects is included, and node-layer edges
    are created between all combinations with probability `p`.

    Args:
        n_layers_per_aspect (list of int): Number of layers in each aspect.
        n_nodes (int): Number of nodes in the network.
        p (float): Probability of including an edge between two node-layer tuples.
        randomWeights (bool): Whether to assign random weights to edges. If False, all weights are 1.
        directed (bool): Whether the network should be directed.

    Returns:
        MultilayerNetwork: A multilayer network instance populated with randomly sampled edges.
    """
    mnet = MultilayerNetwork(
        aspects=len(n_layers_per_aspect),
        directed=directed
    )

    for node in range(n_nodes):
        mnet.add_node(node)
    if not n_layers_per_aspect:
        for aspect in range(len(n_layers_per_aspect)):
            for layer in range(n_layers_per_aspect[aspect]):
                mnet.add_layer(layer, aspect+1)
    
    for edge in __generate_random_edges_er(n_nodes, n_layers_per_aspect, p):
        if randomWeights:
            weight = random.random()
        else:
            weight = 1
        mnet[tuple(edge)] = weight

    
    return mnet

def __generate_random_edges_er(n_nodes, aspect_sizes, p, directed=False):
    """
    Generate random edges for a multilayer network using an Erdős-Rényi process.

    Each node-layer tuple is paired with another with probability `p`, creating multilayer edges.
    Assumes full Cartesian product over all aspects.

    Args:
        n_nodes (int): Number of nodes.
        aspect_sizes (list of int): Number of layers in each aspect.
        p (float): Probability of adding an edge between two node-layer combinations.
        directed (bool): Whether the edges are directed (currently unused).

    Yields:
        tuple: Flattened tuple representing an edge between two node-layer configurations.
    """
    if len(aspect_sizes) > 0:
        total_layers = 1
        for size in aspect_sizes:
            total_layers *= size
        total_node_layers = n_nodes * total_layers

        # Helper: convert flat index to node-layer tuple
        def index_to_node_layer(idx):
            node = idx // total_layers
            rest = idx % total_layers
            layers = []
            for size in reversed(aspect_sizes):
                layers.append(rest % size)
                rest //= size
            return (node, *reversed(layers))

        for i in range(total_node_layers):
            for j in range(i + 1, total_node_layers):
                if random.random() < p:
                    nl1 = index_to_node_layer(i)
                    nl2 = index_to_node_layer(j)
                    yield (nl1[0], nl2[0], *nl1[1:], *nl2[1:])  # Flatten the tuple
    else:
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if random.random() < p:
                    yield (i, j)

def ws_multiplex(
        n_nodes,
        n_layers,
        degrees = [],
        rewiring_probs = [],
) :
    """
    Generate a Watts-Strogatz-like small-world multiplex network.

    Each layer is initialized as a regular ring lattice and edges are randomly rewired
    based on the specified probabilities.

    Args:
        n_nodes (int): Number of nodes in the network.
        n_layers (int): Number of layers (single aspect).
        degrees (list of int): Desired degree (must be even) for each node in each layer.
        rewiring_probs (list of float): Rewiring probabilities per layer.

    Returns:
        tuple:
            MultiplexNetwork: A multiplex network with WS structure in each layer.
            list of tuple: Rewiring operations performed ((src, layer), (old_dst, layer)).
    """

    multplex = MultiplexNetwork(couplings='categorical')
    rewires = [] 
    for i in range(n_nodes):
        multplex.add_node(i)

    for i in range(n_layers):
        multplex.add_layer(i)

    for i in range(n_layers):
        deg = degrees[i]//2
        for j in range(n_nodes):
            for edges in range(deg):
                multplex.A[i][j,(j+edges+1)%n_nodes] = 1
                
    for i in range(n_layers):
        for j in range(n_nodes):
                # iterate on neighbors of node j in layer i
                neighbors = list(multplex._iter_neighbors_out((j,i),(None,i)))
                not_neighbors = _get_not_neighbors_layer(multplex, j, i)

                for neighbor in neighbors:
                    if random.random() < rewiring_probs[i] and len(not_neighbors) > 0:
                        rewired = random.choice(not_neighbors)
                        print(((j,i), neighbor), " to ", ((j,i), (rewired,i)))
                        multplex.A[i][j,neighbor[0]] = 0
                        multplex.A[i][j,rewired] = 1
                        rewire = ((j,i),(neighbor,i))
                        rewires.append(rewire)
                        
    
    return multplex, rewires

def _get_not_neighbors_layer(net, node, layer, loops = False):
    """
    Get a list of nodes in a layer that are not neighbors of a given node.

    Args:
        net (MultiplexNetwork): The multiplex network.
        node (int): Node for which to find non-neighbors.
        layer (int): Layer in which to check for neighbors.
        loops (bool): Whether to include self-loops.

    Returns:
        list of int: Node IDs that are not neighbors of the given node in the specified layer.
    """
    nodelayers = [(x,layer)  for x in list(net.iter_nodes(layer))]
    neighbors = list(net._iter_neighbors((node, layer),(None,layer)))
    not_neighbors = [node for node in nodelayers if node not in neighbors]
    if not loops:
        not_neighbors = [n[0] for n in not_neighbors if n != (node,layer)]
    return not_neighbors


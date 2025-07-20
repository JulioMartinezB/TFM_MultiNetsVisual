import heapq
import numpy as np
import pymnet as net


def dijkstra(self, start, max_distance=100000):
    """
    Compute shortest-path distances from a given node in a multilayer network using Dijkstra's algorithm.

    Args:
        self (MultilayerNetwork): The multilayer network to analyze.
        start (tuple): The starting node (as a multilayer node tuple).
        max_distance (int): Maximum path length to explore.

    Returns:
        dict: A dictionary mapping nodes to their shortest-path distance from the `start` node.
    """
    prio_queue = [(0,start)]  
    distances = { start : 0}
    visited = set()

    while prio_queue:
        distance, node = heapq.heappop(prio_queue)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in self._iter_neighbors(node):
            if type(self) is net.MultiplexNetwork and self._get_degree(neighbor) == len(self.slices):
                continue  # skip if the neighbor is a multiplex node that is not connected to any node 
            if neighbor not in visited and distance+1 <= max_distance and neighbor not in distances:
                heapq.heappush(prio_queue, (distance+1,neighbor))
                distances[neighbor] = distance+ self[self._nodes_to_link(node, neighbor)]
    
    return distances


def layer_connections(self, node):
    """
    Count how many neighbors a node has in each layer.

    Args:
        self (MultilayerNetwork): The multilayer network to analyze.
        node (tuple): The node to inspect.

    Returns:
        dict: A dictionary mapping each layer (tuple of layer identifiers) to the number of neighbors in that layer.
    """

    connections = {}
    for neighbor in self._iter_neighbors(node):
        if type(self) is net.MultiplexNetwork and self._get_degree(neighbor) == len(self.slices):
            continue  # skip if the neighbor is a multiplex node that is not connected to any node 
        layer = neighbor[1:]
        if layer in connections:
            connections[layer] += 1
        else:
            connections[layer] = 1
    return connections


def mu_PCI(self, node, mu=1):
    """
    Compute the μ-PCI of a node: the maximum number k of nodes within μ hops
    that each have a degree ≥ k.

    Args:
        self (MultilayerNetwork): The multilayer network to analyze.
        node (tuple): The node to evaluate.
        mu (int): Maximum path length to consider.

    Returns:
        int: The μ-PCI value for the given node.
    """

    distances = dijkstra(self, node, mu)
    # remove node from distances
    distances.pop(node, None)
    degrees = np.array([self._get_degree(node) for node in distances])
    # sort the degrees descending
    degrees = np.sort(degrees)[::-1]

    # indices = np.arange(1,len(degrees)+1)
    seq = (k for k, d in enumerate(degrees, start=1) if d >= k)
    if not seq:
        mu_PCI = 0
    else:
        mu_PCI = max(seq, default=0)
    return mu_PCI
        
def mlPCI_n(self, node, n=1):
    """
    Compute the ml-PCI_n of a node: the maximum number k of neighbors
    that are connected to at least n different layers with at least k edges.

    Args:
        self (MultilayerNetwork): The multilayer network to analyze.
        node (tuple): The node to evaluate.
        n (int): Number of layers a neighbor must be connected to.

    Returns:
        int: The ml-PCI_n value for the given node.
    """
    #layer_degrees is a numpy array with k rows and n columns, being k the number of neighbors of the node and n the parameter n
    layer_degrees = np.zeros(len([1 for _ in self._iter_neighbors(node)]))
    # for each neighbor of the node, get the number of connections to each layer and store it in layer_degrees
    for i, neighbor in enumerate(self._iter_neighbors(node)):
        if type(self) is net.MultiplexNetwork and self._get_degree(neighbor) == len(self.slices):
            continue  # skip if the neighbor is a multiplex node that is not connected to any node 
        if type(self) is net.MultiplexNetwork and neighbor[0] == node[0]:
            continue  # skip if the neighbor is the same node in a different layer
        layer_conn = layer_connections(self, neighbor)
        layer_conn = sorted(list(layer_conn.values()),reverse=True)
        if len(layer_conn) < n:
            layer_degrees[i] = 0
        else:
            ey = layer_conn[n-1]
            layer_degrees[i] = ey
        
    layer_degrees = sorted(layer_degrees,reverse=True)
    
    # mlPCI_n = np.max(np.where(layer_degrees >= indices, indices, 0))
    seq = (k for k, d in enumerate(layer_degrees, start=1) if d >= k)
    # if seq is empty, return 0
    if not seq:
        mlPCI_n = 0
    else:
    # get the maximum k such that d >= k
        mlPCI_n = max(seq, default=0)


    return mlPCI_n
                    
   
def allPCI(self, node):
    """
    Compute the all-PCI of a node: the maximum number k of neighbors
    that are connected to all layers in the network.

    Args:
        self (MultilayerNetwork): The multilayer network to analyze.
        node (tuple): The node to evaluate.

    Returns:
        int: The all-PCI value for the given node.
    """

    aspects_lens = np.array([len(self.slices[i]) for i in range(1,self.aspects+1)])
    n_layers = aspects_lens.prod()
    all_PCI = 0
    for neighbor in self._iter_neighbors(node):
        if type(self) is net.MultiplexNetwork and self._get_degree(neighbor) == len(self.slices):
            continue
        if type(self) is net.MultiplexNetwork and neighbor[0] == node[0]:
            continue
        layer_conn = layer_connections(self, neighbor)
        if len(layer_conn) < n_layers:
            continue
        all_PCI += 1
        
        
    return all_PCI 


def lsPCI(self, node):
    """
    Compute the ls-PCI of a node: the maximum number k such that the node
    has at least k neighbors, each connected to at least k different nodes
    in at least k different layers.

    Args:
        self (MultilayerNetwork): The multilayer network to analyze.
        node (tuple): The node to evaluate.

    Returns:
        int: The ls-PCI value for the given node.
    """

    neighbors_layer_degrees = []
    for neighbor in self._iter_neighbors(node):
        if type(self) is net.MultiplexNetwork and self._get_degree(neighbor) == len(self.slices):
            continue  # skip if the neighbor is a multiplex node that is not connected to any node 
        if type(self) is net.MultiplexNetwork and neighbor[0] == node[0]:
            continue
        layer_conn = layer_connections(self, neighbor)
        neighbors_layer_degrees.append(sorted(list(layer_conn.values()),reverse=True))
    dim2_seq = [len(layer_conn) for layer_conn in neighbors_layer_degrees]
    if len(dim2_seq) == 0:
        return 0
    

    nl_matrix = np.zeros((len(neighbors_layer_degrees),max([len(layer_conn) for layer_conn in neighbors_layer_degrees])))
    for i, layer_conn in enumerate(neighbors_layer_degrees):
        nl_matrix[i,:len(layer_conn)] = layer_conn

    lsPCI = 0
    for i in range(0,nl_matrix.shape[1]):
        discarded = np.where(nl_matrix[:,i] < i)[0]
        if len(discarded) > i:
            lsPCI = i
            break
    return lsPCI
    
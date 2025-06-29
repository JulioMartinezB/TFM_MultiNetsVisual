from pymnet import MultilayerNetwork
from collections import defaultdict



def modularity(communities):
    """
    Compute the modularity score for a given community structure.

    Args:
        communities (dict): Community metadata with the following keys:
            - com2nodes (dict): Mapping from community ID to list of node-layer tuples.
            - com_inner_weight (dict): Sum of edge weights within each community.
            - com_total_weight (dict): Total edge weight incident to each community.
            - graph_size (float): Total edge weight in the graph.

    Returns:
        float: Modularity value (higher is better).
    """

    mod = 0
    graph_size = communities.get('graph_size', 0)
    for community in communities['com2nodes'].keys():
        comm_inner = communities['com_inner_weight'][community]
        comm_total = communities['com_total_weight'][community]
        mod += comm_inner/(2*graph_size) - (comm_total/(2*graph_size))**2
    return mod


def merge_coms(coms, com1, com2, weigth_inter_coms):
    """
    Merge two communities and update the statistics.

    Args:
        coms (dict): Original community structure.
        com1 (hashable): Target community ID.
        com2 (hashable): Community ID to merge into com1.
        weigth_inter_coms (float): Total weight of edges between com1 and com2.

    Returns:
        dict: Updated community structure after merging.
    """
  
    new_coms = coms.copy()
    new_coms['com2nodes'][com1] = new_coms['com2nodes'][com1] + new_coms['com2nodes'][com2]

    for node in new_coms['com2nodes'][com2]:
        new_coms['node2com'][node] = com1
        
    # weigth_inter_coms duplicated as those edges become edges between a node and itself  
    new_coms['com_inner_weight'][com1] += new_coms['com_inner_weight'][com2] + 2*weigth_inter_coms
    new_coms['com_total_weight'][com1] += new_coms['com_total_weight'][com2] 

    new_coms['com2nodes'].pop(com2)
    new_coms['com_inner_weight'].pop(com2)
    new_coms['com_total_weight'].pop(com2)

    return new_coms

def weights_inter_coms(net, coms):
    """
    Calculate the weights of edges between all pairs of different communities.

    Args:
        net (MultilayerNetwork): The network to analyze.
        coms (dict): Community assignment structure.

    Returns:
        dict: Mapping from (com1, com2) to total weight of edges between them.
    """    
    inter_coms = defaultdict(int)
    for edge in net.edges:
        # last value in edge is the weight, so we ignore it
        edge = tuple(list(edge)[:-1])
        n1,n2 = net._link_to_nodes(edge)
        com1 = coms['node2com'][n1]
        com2 = coms['node2com'][n2]
        if com1 != com2:
            inter_coms[(min(com1, com2), max(com1, com2))] += net[n1][n2]
    return inter_coms



def louvain_step(net, communities):
    """
    Perform one step of the Louvain algorithm by merging the best pair of communities.

    Args:
        net (MultilayerNetwork): The network under analysis.
        communities (dict): Current community structure.

    Returns:
        dict: Updated community structure after merging, if beneficial.
    """
    com_keys = list(communities['com2nodes'].keys())
    n_coms = len(com_keys)
    max_objective_value = communities['objective_function_value']
    weights = weights_inter_coms(net,communities)
    
    max_coms = communities

    for i in range(n_coms):
        for j in range(i+1,n_coms):

            c1, c2 = com_keys[i], com_keys[j]
            if (c1, c2) not in weights and (c2, c1) not in weights:
                continue
            w_inter = weights.get((c1, c2), weights.get((c2, c1), 0))

            temp_coms = {
                'com2nodes': {k: v[:] for k, v in communities['com2nodes'].items()},
                'node2com' : communities['node2com'].copy(),
                'com_inner_weight': communities['com_inner_weight'].copy(),
                'com_total_weight': communities['com_total_weight'].copy(),
                'graph_size': communities['graph_size'],
                'objective_function': communities['objective_function'],
                'objective_function_name': communities['objective_function_name'],
                'objective_function_value': communities['objective_function_value'],
            }

            # print(com_keys[i],com_keys[j])
            merged_coms = merge_coms(temp_coms,com_keys[i],com_keys[j],weights[(com_keys[i],com_keys[j])])
            new_objective_value = merged_coms['objective_function'](merged_coms)
            merged_coms['objective_function_value'] = new_objective_value
            
            
            if new_objective_value > max_objective_value:
                max_objective_value = new_objective_value
                max_coms = merged_coms
    return max_coms


def louvain_algorithm(net, obj_function=modularity, max_iter=100):
    """
    Execute the full Louvain algorithm on a multilayer network.

    Args:
        net (MultilayerNetwork): The multilayer network to cluster.
        obj_function (callable): Objective function to optimize (e.g., modularity).
        max_iter (int): Maximum number of iterations.

    Returns:
        list of dict: History of community structures per iteration.
    """

    communities = init_multiplex_communities_louvain(net, obj_function)
    states = [communities]
    for _ in range(max_iter):
        new_communities = louvain_step(net, communities)
        if new_communities['objective_function_value'] <= communities['objective_function_value']:
            break
        states.append(new_communities)
        communities = new_communities
    
    return states     


def init_multiplex_communities_louvain(net, obj_function=modularity):
    """
    Initialize each physical node as its own community.

    Args:
        net (MultilayerNetwork): Network to initialize.
        obj_function (callable): Objective function to evaluate communities.

    Returns:
        dict: Initial community structure with weights and mappings.
    """
    # all representation of a node is within the same community
    coms = list(net.slices[0])

    communities = {
        'com2nodes' : {com : [nl for nl in net.iter_node_layers() if nl[0] == com] for com in coms},
        'node2com' : {},
        'com_inner_weight' : {com : 0 for com in coms},
        'com_total_weight' : {com : 0 for com in coms},
        'graph_size' : len(net.edges),
        'objective_function' : obj_function,
        'objective_function_name' : obj_function.__name__,
        'objective_function_value' : 0,
    }


    for com, nodes in communities['com2nodes'].items():
        inner_w = 0
        total_w = 0
        for node1 in nodes:
            for node2 in nodes:
                inner_w += net[node1][node2]
            for neighbor in net._iter_neighbors(node1):
                total_w += net[node1][neighbor]
        communities['com_inner_weight'][com] = inner_w
        communities['com_total_weight'][com] = total_w
    
    communities['node2com'] = {node: com for com, nodes in communities['com2nodes'].items() for node in nodes}
    communities['objective_function_value'] = obj_function(communities)
    return communities    




##### new net generation functions #####

def communities_are_neighbors(net, com1, com2, communities):
    """
    Determine if two communities are adjacent in the network.

    Args:
        net (MultilayerNetwork): Network to inspect.
        com1 (hashable): First community ID.
        com2 (hashable): Second community ID.
        communities (dict): Current community assignments.

    Returns:
        bool: True if any node in com1 is linked to any node in com2.
    """

    
    for node1 in communities['com2nodes'][com1]:
        for node2 in net._iter_neighbors(node1):
            if node2 in communities['com2nodes'][com2]:
                return True
    return False

def get_highest_degree_node(net, communities, com):
    """
    Identify the physical node with the highest total degree in a community.

    Args:
        net (MultilayerNetwork): Network under analysis.
        communities (dict): Community structure.
        com (hashable): Community ID.

    Returns:
        hashable: Physical node identifier with highest degree.
    """


    nodes_no_layers_degree = { n_l[0] : 0 for n_l in communities['com2nodes'][com]}

    for node1 in communities['com2nodes'][com]:
        nodes_no_layers_degree[node1[0]] += net._get_degree(node1)
    # get the node with the highest degree
    highest_degree_node = max(nodes_no_layers_degree, key=nodes_no_layers_degree.get)
    return highest_degree_node
        
def generate_louvain_communities_net(net, communities):
    """
    Construct a new multilayer network where each node represents a community.

    Args:
        net (MultilayerNetwork): Original multilayer network.
        communities (dict): Final community structure after Louvain algorithm.

    Returns:
        tuple:
            - MultilayerNetwork: Network where nodes represent communities.
            - dict: Mapping from community node to its size (number of elements).
            - dict: Mapping from original community ID to representative node.
    """

    
    com_net = MultilayerNetwork(aspects=1, directed=False)
    
    com_2_name = {com:"" for com in communities['com2nodes'].keys()}
    for com, nodes in communities['com2nodes'].items():
        com_2_name[com] = get_highest_degree_node(net, communities, com)


    for com in communities['com2nodes'].keys():
        com_net.add_node(com_2_name[com])

    neighbors = defaultdict(set)
    for com in communities['com2nodes'].keys():
        for com2 in communities['com2nodes'].keys():
            if com != com2 and communities_are_neighbors(net, com, com2, communities):
                com_name = com_2_name[com]
                com2_name = com_2_name[com2]
                neighbors[com_name].add(com2_name)
            if com2 != com and communities_are_neighbors(net, com2, com, communities):
                com_name = com_2_name[com]
                com2_name = com_2_name[com2]
                neighbors[com2_name].add(com_name)

    for com, neighbors_set in neighbors.items():
        for neighbor in neighbors_set:
            com_net[(com,1)][(neighbor,1)] = 1  # or any weight you want, here we use 1

    com_sizes = {com_2_name[com]: len(nodes) for com, nodes in communities['com2nodes'].items()}

    

    return com_net, com_sizes, com_2_name

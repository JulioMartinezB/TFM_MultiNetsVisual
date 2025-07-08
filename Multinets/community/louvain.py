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



def move_nodes_step(self, subcoms, force_merge=False):
    """
    Perform a step of the Louvain algorithm by moving nodes between communities.

    Parameters
    ----------
    self : MultilayerNetwork
        The multilayer network to analyze.
    subcoms : dict
        Current communities structure.
    force_merge : bool, optional
        If True, forces the merging of two communities with the smallest modularity loss
        when no improvement is found. Default is False.

    Returns
    -------
    dict
        Updated communities structure.
    """
    sub_coms = list(subcoms['com2nodes'].keys())
    new_coms = {
        'com2nodes': {k: v[:] for k, v in subcoms['com2nodes'].items()},
        'node2com': subcoms['node2com'].copy(),
        'com_inner_weight': subcoms['com_inner_weight'].copy(),
        'com_total_weight': subcoms['com_total_weight'].copy(),
        'neigh_coms': {k: set(v) for k, v in subcoms['neigh_coms'].items()},
        'graph_size': subcoms['graph_size'],
        'objective_function_value': subcoms['objective_function_value'],
    }
    improvement = True
    already_forced_merge = False 
    iteration = 0

    while improvement and not (already_forced_merge and force_merge):
        improvement = False
        iteration += 1
        random.shuffle(sub_coms)

        best_merge_coms = None
        best_merge_gain = float('-inf')

        for subcom in sub_coms:
            node_in_subcom = subcoms['com2nodes'][subcom][0]
            current_com = new_coms['node2com'][node_in_subcom]
            neighbor_coms = new_coms['neigh_coms'][current_com]

            best_increase = 0
            best_coms = new_coms

            for target_com in neighbor_coms:
                temp_coms = {
                    'com2nodes': {k: v[:] for k, v in new_coms['com2nodes'].items()},
                    'node2com': new_coms['node2com'].copy(),
                    'com_inner_weight': new_coms['com_inner_weight'].copy(),
                    'com_total_weight': new_coms['com_total_weight'].copy(),
                    'neigh_coms': {k: set(v) for k, v in new_coms['neigh_coms'].items()},
                    'graph_size': new_coms['graph_size'],
                    'objective_function_value': new_coms['objective_function_value'],
                }

                temp_coms = move_subcom(self, subcoms, temp_coms, subcom, current_com, target_com)
                gain = temp_coms['objective_function_value'] - new_coms['objective_function_value']

                if gain > best_increase:
                    best_increase = gain
                    best_coms = temp_coms

                # Track the best merge with the smallest gain
                if gain > best_merge_gain:
                    best_merge_gain = gain
                    best_merge_coms = temp_coms

            if best_increase > 0:
                improvement = True
                new_coms = best_coms

        # Force merge if no improvement and force_merge is enabled
        if not improvement and force_merge and best_merge_coms:
            already_forced_merge = True
            new_coms = best_merge_coms

    return new_coms

def move_subcom(self, subcoms, coms, subcom, current_com, target_com):
    """
    Move a subcommunity to a target community and update the communities structure.
    
    Parameters
    ----------
    self : MultilayerNetwork
        The multilayer network to analyze.
    subcoms : list
        List of subcommunities.
    coms : dict
        Current communities structure.
    subcom : str
        The subcommunity to move.
    target_com : str
        The target community to move the subcommunity to.
    
    Returns
    -------
    dict
        Updated communities structure after moving the subcommunity.
    """

    prev_modularity_target = coms['com_inner_weight'][target_com] / (2*coms['graph_size']) - (coms['com_total_weight'][target_com] / (2*coms['graph_size']))**2
    prev_modularity_current = coms['com_inner_weight'][current_com] / (2*coms['graph_size']) - (coms['com_total_weight'][current_com] / (2*coms['graph_size']))**2

    new_coms = coms.copy()
    #move nodes of the subcommunity to the target community
    new_coms['com2nodes'][target_com].extend(subcoms['com2nodes'][subcom])
    for node in subcoms['com2nodes'][subcom]:
        new_coms['com2nodes'][current_com].remove(node)


    # update node2com mapping
    for node in subcoms['com2nodes'][subcom]:
        new_coms['node2com'][node] = target_com

    # update total weights
    new_coms['com_total_weight'][target_com] = 0
    new_coms['com_total_weight'][current_com] = 0

    new_coms['com_inner_weight'][target_com] = 0
    new_coms['com_inner_weight'][current_com] = 0

    new_coms['neigh_coms'][target_com] = set()
    new_coms['neigh_coms'][current_com] = set()

    for node in new_coms['com2nodes'][target_com]:
        # update the neighbors of the target community
        for neighbor in self._iter_neighbors(node):
            new_coms['com_total_weight'][target_com] += self[node][neighbor]
            if new_coms['node2com'][neighbor] != target_com:
                new_coms['neigh_coms'][target_com].add(new_coms['node2com'][neighbor])
            else:
                new_coms['com_inner_weight'][target_com] += self[node][neighbor]
    for node in new_coms['com2nodes'][current_com]:
        # update the neighbors of the current community
        for neighbor in self._iter_neighbors(node):
            new_coms['com_total_weight'][current_com] += self[node][neighbor]
            if new_coms['node2com'][neighbor] != current_com:
                new_coms['neigh_coms'][current_com].add(new_coms['node2com'][neighbor])
            else:
                new_coms['com_inner_weight'][current_com] += self[node][neighbor]

    
    new_modularity_target = new_coms['com_inner_weight'][target_com] / (2*new_coms['graph_size']) - (new_coms['com_total_weight'][target_com] / (2*new_coms['graph_size']))**2
    new_modularity_current = new_coms['com_inner_weight'][current_com] / (2*new_coms['graph_size']) - (new_coms['com_total_weight'][current_com] / (2*new_coms['graph_size']))**2


    if len(new_coms['com2nodes'][current_com]) == 0:
        # remove the current community if it has no nodes left
        new_coms['com2nodes'].pop(current_com)
        new_coms['com_inner_weight'].pop(current_com)
        new_coms['com_total_weight'].pop(current_com)
        for k in new_coms['neigh_coms'].keys():
            if current_com in new_coms['neigh_coms'][k] and k != current_com:
                new_coms['neigh_coms'][k].remove(current_com)
                new_coms['neigh_coms'][k].add(target_com)
        new_coms['neigh_coms'].pop(current_com)
                                                                  
    # update modularity
    new_coms['objective_function_value'] =  new_coms['objective_function_value'] - prev_modularity_current - prev_modularity_target + new_modularity_target + new_modularity_current


    return new_coms


def louvain_algorithm(net, max_iter=1000, force_merge=False):
    """
    Apply the Louvain algorithm to find communities in a multiplex network.
    
    Parameters
    ----------
    net : MultilayerNetwork
        The multilayer network to analyze.
    max_iter : int, optional
        The maximum number of iterations to run the algorithm (default is 1000).
    force_merge : bool, optional
        If True, forces the merging of communities with the smallest modularity loss when no improvement is found.
    
    Returns
    -------
    dict
        A dictionary containing the communities and their properties.
    """
    
    communities = init_multiplex_communities_louvain(net)
    states = [communities]
    for _ in range(max_iter):
        print(f"Iteration {_+1}/{max_iter}")
        new_communities = move_nodes_step(net, communities, force_merge=force_merge)
        if len(new_communities['com2nodes']) == len(communities['com2nodes']):
            # if there is no improvement, we stop
            print("No movements found, stopping.")
            states.append(new_communities)
            break
        if len(new_communities['com2nodes']) == 1:
            # if there is only one community, we stop
            states.append(new_communities)
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

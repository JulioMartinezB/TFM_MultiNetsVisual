import random
import copy
import queue
import math
from pymnet import MultilayerNetwork
from collections import defaultdict





def leiden_algorithm(net, gamma=1.0, max_iterations=100):
    """
    Perform the Leiden algorithm for community detection in a multilayer network.

    This algorithm identifies communities through iterative improvement based on the Constant Potts Model (CPM).

    Args:
        net (MultilayerNetwork): The multilayer network to analyze.
        gamma (float, optional): Resolution parameter. Higher values lead to smaller communities.
        max_iterations (int, optional): Maximum number of iterations to run. Default is 100.

    Returns:
        list of dict: List of community states for each iteration until convergence.
    """

    
    communities = init_multiplex_communities_leiden(net)
    states = [communities]
    for _ in range(max_iterations):
        old_communities, communities, merge_history = leiden_FastNodeMove(net, communities, gamma)
        communities = refine_partition(net, communities,old_communities, merge_history, gamma)

        # check if the com2nodes of old_communities and communities are the same
        if old_communities['com2nodes'] == communities['com2nodes']:
            print("Convergence reached.")
            break
        else:
            states.append(communities)


    return states


def init_multiplex_communities_leiden(net):
    """
    Initialize each node-layer pair in its own community for the Leiden algorithm.

    Args:
        net (MultilayerNetwork): The network to initialize communities on.

    Returns:
        dict: A dictionary with initial community structures including:
            - com2nodes: Community to node-layer mapping.
            - node2com: Node-layer to community mapping.
            - com_inner_weight: Intra-community weights.
            - com_total_weight: Total weights per community.
            - neigh_coms: Neighboring community relationships.
            - graph_size: Total number of edges.
    """

    # all representation of a node is within the same community
    coms = list(net.slices[0])

    communities = {
        'com2nodes' : {com : [nl for nl in net.iter_node_layers() if nl[0] == com] for com in coms},
        'node2com' : {},
        'com_inner_weight' : {com : 0 for com in coms},
        'com_total_weight' : {com : 0 for com in coms},
        'graph_size' : len(net.edges),
        'neigh_coms' : {com : set() for com in coms},

    }


    for com, nodes in communities['com2nodes'].items():
        inner_w = 0
        total_w = 0
        for node1 in nodes:
            for node2 in nodes:
                inner_w += net[node1][node2]
            for neighbor in net._iter_neighbors(node1):
                total_w += net[node1][neighbor]
            communities['neigh_coms'][com].update([x[0] for x in net._iter_neighbors(node1) if x[0] != node1[0]])

        communities['com_inner_weight'][com] = inner_w
        communities['com_total_weight'][com] = total_w
    
    communities['node2com'] = {node: com for com, nodes in communities['com2nodes'].items() for node in nodes}
    return communities

def leiden_FastNodeMove(net, communities, gamma):
    """
    Perform the node movement and merging phase of the Leiden algorithm.

    Args:
        net (MultilayerNetwork): The multilayer network.
        communities (dict): Current community structure.
        gamma (float): Resolution parameter.

    Returns:
        tuple: (old_communities, new_communities, merge_history)
            - old_communities: Previous state before moves.
            - new_communities: Updated community structure.
            - merge_history: History of merges performed.
    """
    q = queue.Queue()
    
    coms = list(communities['com2nodes'].keys())
    old_communities = copy.deepcopy(communities)

    random.shuffle(coms)
    for com in coms:
        q.put(com)
    # print(q.qsize())
    elems_in_q = coms

    merge_history = {com: [com] for com in coms}
    
    while not q.empty():
        com1 = q.get()
        elems_in_q.remove(com1)
        # print("step: ", com1)
        if com1 not in communities['com2nodes']:            
            continue
        max_cpm = 0
        max_com = com1
        for com2 in communities['neigh_coms'][com1]:
            weights_inter_coms = weights_inter_com1_com2(net,communities,com1,com2)
            new_cpm=compute_inc_CPM(communities,com1,com2,weights_inter_coms,gamma)
            # print("  com1: ",com1," com2: ",com2," new_cpm: ",new_cpm)
            if new_cpm > max_cpm:
                max_com = com2
                max_cpm = new_cpm
        if max_cpm > 0: 
            # print("  merging with",max_com)
            # print("  max_cpm ",max_cpm)
            weights_inter_coms = weights_inter_com1_com2(net,communities,com1,max_com)
            old_neigh_com1 = communities['neigh_coms'][com1]
            communities = merge_communities(net,communities,com1,max_com,weights_inter_coms)
            
            #append the merge history of max_com to com1
            merge_history[com1] = merge_history[com1] + merge_history[max_com]
            merge_history.pop(max_com, None)  # remove max_com from history

            

            for com in old_neigh_com1:
                if com not in elems_in_q:
                    q.put(com)   
                    elems_in_q.append(com)

    return old_communities, communities, merge_history

def refine_partition(net, parent_coms, subcoms, merge_history, gamma):
    """
    Refine community partitions by evaluating interconnections among subcommunities.

    Args:
        net (MultilayerNetwork): The multilayer network.
        parent_coms (dict): Original community structure before refinement.
        subcoms (dict): Community structure after merges.
        merge_history (dict): Tracks how subcommunities were merged.
        gamma (float): Resolution parameter.

    Returns:
        dict: Refined community structure.
    """
    for com_merged, subcoms_merged in merge_history.items():
        # print("Refining partition for com_merged:", com_merged, "with subcoms_merged:", subcoms_merged)
        R = check_interconected_subcoms(net, parent_coms, subcoms, com_merged, subcoms_merged, gamma)
        singleton = R.copy()
        # print("R: ", R)
        if len(R) < 2:
            # print("Skipping refinement for com_merged:", com_merged, "as R has less than 2 elements.")
            continue
        for subcom1 in R:
            # check if subcom1 is a singleton using singleton list
            if subcom1 not in singleton:
                continue


            T = check_interconected_subcoms(net, parent_coms, subcoms, com_merged, subcoms_merged, gamma)
            # print("     T: ", T)
            if not T:
                continue
            

            # Compute ΔH for each target subcommunity
            inc_CPMs = []
            for subcom2 in T:
                if subcom1 == subcom2:
                    continue
                weights_com1_com2 = weights_inter_com1_com2(net, subcoms, subcom1, subcom2)
                inc_CPM = compute_inc_CPM(subcoms, subcom1, subcom2,weights_com1_com2, gamma)
                inc_CPMs.append((subcom2, inc_CPM))

                        # Softmax-based random selection
            probs = []
            for (subcom2, inc_CPM) in inc_CPMs:
                # print("         subcom1:", subcom1, "subcom2:", subcom2, "inc_CPM:", inc_CPM)
                if inc_CPM > 0:
                    # print(math.exp(0.5 * inc_CPM))
                    probs.append(math.exp(0.5 * inc_CPM))
                else:
                    probs.append(0.0)

            if sum(probs) == 0:
                continue

            chosen = random.choices([x[0] for x in inc_CPMs], weights=probs, k=1)[0]

            if subcom1 in singleton:
                singleton.remove(subcom1)
            if chosen in singleton:
                singleton.remove(chosen)
            # print("         singleton: ", singleton)

            # print('         merging subcom1:', subcom1, 'with subcom2:', chosen, 'with inc_CPM:', inc_CPMs)
            subcoms = merge_communities(net, subcoms, chosen, subcom1, 
                                             weights_inter_com1_com2(net, subcoms, chosen, subcom1))

            subcoms_merged.remove(subcom1)


    return subcoms




def get_highest_degree_node(net, communities, com):
    """
    Identify the node (ignoring layers) with the highest degree in a community.

    Args:
        net (MultilayerNetwork): The multilayer network.
        communities (dict): Current community structure.
        com (str): Community ID.

    Returns:
        str: Node name with the highest aggregated degree.
    """


    nodes_no_layers_degree = { n_l[0] : 0 for n_l in communities['com2nodes'][com]}

    for node1 in communities['com2nodes'][com]:
        nodes_no_layers_degree[node1[0]] += net._get_degree(node1)
    # get the node with the highest degree
    highest_degree_node = max(nodes_no_layers_degree, key=nodes_no_layers_degree.get)
    return highest_degree_node
        
def weights_inter_com1_com2(net, coms, com1, com2):
    """
    Compute total weight of edges between two communities.

    Args:
        net (MultilayerNetwork): The network.
        coms (dict): Community dictionary with com2nodes and node2com.
        com1 (str): ID of the first community.
        com2 (str): ID of the second community.

    Returns:
        float: Sum of weights between `com1` and `com2`.
    """
    weight_inter_coms = 0
    for node1 in coms['com2nodes'][com1]:
        for neighbor in net._iter_neighbors(node1):
            if coms['node2com'][neighbor] == com2:
                weight_inter_coms += net[node1][neighbor]
    return weight_inter_coms

def weights_inter_subcom(net, subcoms, subcoms_merged):
    """
    Compute pairwise edge weights between all subcommunities in a set.

    Args:
        net (MultilayerNetwork): The network.
        subcoms (dict): Community dictionary.
        subcoms_merged (list): List of subcommunity IDs.

    Returns:
        dict: Mapping from subcommunity ID to total inter-subcommunity weight.
    """
    edge_weights = { subcom: 0 for subcom in subcoms_merged }
    for i in range(len(subcoms_merged)):
        subcom = subcoms_merged[i]
        for j in range(i + 1, len(subcoms_merged)):
            other_subcom = subcoms_merged[j]
            if subcom == other_subcom:
                continue
            weight = weights_inter_com1_com2(net, subcoms, subcom, other_subcom)
            edge_weights[subcom] += weight
            edge_weights[other_subcom] += weight
    return edge_weights
                        
def compute_inc_CPM(communities, com1, com2, weights_inter_com1_com2, gamma):
    """
    Compute the change in CPM objective function if two communities are merged.

    Args:
        communities (dict): Current community structure.
        com1 (str): First community ID.
        com2 (str): Second community ID.
        weights_inter_com1_com2 (float): Weight between the two communities.
        gamma (float): Resolution parameter.

    Returns:
        float: Change in CPM score after merging.
    """
    com1_inner = communities['com_inner_weight'][com1]
    com1_size = len(communities['com2nodes'][com1])
    com2_inner = communities['com_inner_weight'][com2]
    com2_size = len(communities['com2nodes'][com2])
    dec = (com1_inner - gamma*(com1_size*(com1_size-1)/2)) + (com2_inner - gamma*(com2_size*(com2_size-1)/2))
    inc = (com1_inner + com2_inner + weights_inter_com1_com2) - gamma*(com1_size + com2_size)*(com1_size + com2_size - 1)/2
    return inc - dec

def merge_communities(net, communities, com1, com2, weight_inter_coms):
    """
    Merge two communities into one, updating structure and metadata.

    Args:
        net (MultilayerNetwork): The network.
        communities (dict): Current community state.
        com1 (str): Community to merge into.
        com2 (str): Community to merge from.
        weight_inter_coms (float): Total weight between the two communities.

    Returns:
        dict: Updated community dictionary.
    """
    communities = communities.copy()
    # merge both communities into the new one
    communities['com2nodes'][com1] = communities['com2nodes'][com1] + communities['com2nodes'][com2]
    # change all nodes to the new community
    for node in communities['com2nodes'][com2]:
        communities['node2com'][node] = com1
    

    # the new internal weight is the sum of both plus twice the sum between the communities
    communities['com_inner_weight'][com1] += communities['com_inner_weight'][com2] + 2*weight_inter_coms

    # new neighboors
    communities['neigh_coms'][com1] = communities['neigh_coms'][com1].union(communities['neigh_coms'][com2])
    communities['neigh_coms'][com1].discard(com1)  # remove self-loop if exists
    communities['neigh_coms'][com1].discard(com2)  # remove self-loop if exists

    communities['com_total_weight'][com1] = communities['com_total_weight'][com1] + communities['com_total_weight'][com2] + weight_inter_coms

    #remove the old community
    communities['com2nodes'].pop(com2)
    communities['com_inner_weight'].pop(com2)
    communities['com_total_weight'].pop(com2)
    communities['neigh_coms'].pop(com2)

    # update the neigh_coms of the neighbors, change com2 for com1 in the neigh_coms for other communities
    for k,v in communities['neigh_coms'].items():
        if com2 in v:
            v.remove(com2)
            v.add(com1)
    # if com1 in communities['neigh_coms'][com1]:
    #     communities['neigh_coms'][com1].remove(com1)

    communities['objective_function_value'] = communities['objective_function'](communities)

    return communities
        
def check_interconected_subcoms(net, parent_coms, subcoms, parent_com, subcoms_in_parent, gamma):
    """
    Determine which subcommunities in a parent are densely connected enough to merge.

    Args:
        net (MultilayerNetwork): The network.
        parent_coms (dict): Pre-refinement community structure.
        subcoms (dict): Post-merge structure.
        parent_com (str): Parent community being refined.
        subcoms_in_parent (list): Subcommunities within the parent.
        gamma (float): Resolution parameter.

    Returns:
        list: Subcommunity IDs eligible for merging.
    """
    # print(parent_com, subcoms_in_parent)
    weights_inter_subcoms = weights_inter_subcom(net, subcoms, subcoms_in_parent)
    R = []
    for subcom1 in subcoms_in_parent:

        e_v_S = weights_inter_subcoms[subcom1]
        # norm_v = deg of all nodes in subcom1
        norm_v = len(subcoms['com2nodes'][subcom1])
        norm_S = len(parent_coms['com2nodes'][parent_com])
        # edges(subcom1, com - subcom1) >= gamma * ||subcom1|| * (||com|| - ||subcom1||) 
        if e_v_S >= gamma * norm_v * (norm_S - norm_v):
            R.append(subcom1)
            continue
    random.shuffle(R)
    return R


def generate_leiden_communities_net(net, communities):
    """
    Generate a new simplified network from detected communities.

    Each node in the output represents a community, and edges reflect inter-community links.

    Args:
        net (MultilayerNetwork): Original network.
        communities (dict): Detected communities from Leiden.

    Returns:
        tuple:
            - MultilayerNetwork: Community-level network.
            - dict: Sizes of each community.
            - dict: Mapping from community ID to representative node.
    """
    
    com_net = MultilayerNetwork(aspects=1, directed=False)


    
    com_2_name = {com:"" for com in communities['com2nodes'].keys()}
    for com, nodes in communities['com2nodes'].items():
        com_2_name[com] = get_highest_degree_node(net, communities, com)


    for com in communities['com2nodes'].keys():
        com_net.add_node(com_2_name[com])

    neighbors = defaultdict(set)
    for com in communities['com2nodes'].keys():
        for com2 in communities['neigh_coms'][com]:
            if com != com2:
                com_name = com_2_name[com]
                com2_name = com_2_name[com2]
                neighbors[com_name].add(com2_name)
    for com, neighbors_set in neighbors.items():
        for neighbor in neighbors_set:
            com_net[(com,1)][(neighbor,1)] = 1  # or any weight you want, here we use 1

    com_sizes = {com_2_name[com]: len(nodes) for com, nodes in communities['com2nodes'].items()}

    return com_net, com_sizes, com_2_name
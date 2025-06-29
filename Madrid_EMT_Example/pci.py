
from Multinets.centrality.pci import mu_PCI, mlPCI_n, allPCI, lsPCI
from Multinets.centrality.eigen import (
    independent_layer_eigenvector_centrality,
    uniform_eigenvector_centrality,
    local_heterogeneus_eigenvector_centrality,
    global_heterogeneus_eigenvector_centrality
)
from Madrid_EMT_Example.utils import read_multiplex_edges_from_csv,EDGES_NO_WEIGHT_PATH



mnet_no_weighted = read_multiplex_edges_from_csv(EDGES_NO_WEIGHT_PATH, weighted=False)

print(mu_PCI(mnet_no_weighted, ('AVENIDA DE AMERICA', 'metro'), mu = 1))
print(mu_PCI(mnet_no_weighted, ('AVENIDA DE AMERICA', 'metro'), mu = 2))

# for each node in the mnet_no_weightedwork store the mu_PCI, mlPCI_n, allPCI and lsPCI in a dictionary
nl_PCIs = {}
for node in mnet_no_weighted.iter_node_layers():
    nl_PCIs[node] = {
        'mu_PCI': mu_PCI(mnet_no_weighted, node, mu=1),
        'mlPCI_n': mlPCI_n(mnet_no_weighted, node),
        'allPCI': allPCI(mnet_no_weighted, node),
        'lsPCI': lsPCI(mnet_no_weighted, node)
    }


# show nodes with lsPCI > 1
high_lsPCI_nodes = {node: pci['lsPCI'] for node, pci in nl_PCIs.items() if pci['lsPCI'] > 1}
print(f"{len(high_lsPCI_nodes)} Nodes with lsPCI > 1:")
# for node, lsPCI in high_lsPCI_nodes.items():
#     print(f"{node}: {lsPCI}"


###################

ILEC_madrid, ILEC_int2layers = independent_layer_eigenvector_centrality(mnet_no_weighted)
UEC_madrid = uniform_eigenvector_centrality(mnet_no_weighted)
LHEC_madrid, LHEC_nodes2int, LHEC_layers2int, LHEC_int2nodes, LHEC_int2layers  = local_heterogeneus_eigenvector_centrality(mnet_no_weighted)
GHEC_madrid, GHEC_nodes2int, GHEC_layers2int, GHEC_int2nodes, GHEC_int2layers = global_heterogeneus_eigenvector_centrality(mnet_no_weighted)

print("ILEC_madrid:", ILEC_madrid.shape)
print("UEC_madrid:", UEC_madrid.shape)
print("LHEC_madrid:", LHEC_madrid.shape)
print("GHEC_madrid:", GHEC_madrid.shape)


# NODE = 'AVENIDA DE AMERICA'
NODE = 'PLAZA ELIPTICA'
LAYER = 'urban'
print(GHEC_madrid[GHEC_layers2int[0][LAYER]][GHEC_nodes2int[NODE]])

print(LHEC_madrid[LHEC_layers2int[0][LAYER]][LHEC_nodes2int[NODE]])
import numpy as np
import tensorflow as tf
from pymnet import MultilayerNetwork
from Multinets.utils_tensorflow import __get_sparse_tensor__, __get_sparse_tensor_between_layers__,\
                                     __slice_sparse_tensor__, __sparse_masked_tensordot, \
                                        khatri_rao_weighted_block_sparse, sparse_power_iteration


def independent_layer_eigenvector_centrality(self):
    """
    Compute the eigenvector centrality independently for each layer in the network.

    Each layer is extracted using a sparse tensor slice, and eigenvector centrality is computed
    using the power iteration method.

    Returns:
        Tuple[tf.Tensor, Dict[int, Any]]: 
            - A tensor of shape [n_layers, n_nodes] containing the eigenvector centrality values per layer.
            - A mapping from integer layer indices to original layer identifiers.
    """    # if tyupe of layers is not int, map them to int
    layers_to_int = {layer: i for i, layer in enumerate(self.get_layers())}
    int_to_layers = {i: layer for i, layer in enumerate(self.get_layers())}
    layers = [layers_to_int[layer] for layer in self.get_layers()]
    layers = tf.convert_to_tensor(layers, dtype=tf.int32)
    # layers = tf.convert_to_tensor(list(self.get_layers()), dtype=tf.int32)

    def map_fn_body(li):
        # li is a scalar tensor (layer index)
        layer_matrix = __get_sparse_tensor_between_layers__(self, tf.expand_dims(li, 0), tf.expand_dims(li, 0))
        layer_matrix = tf.sparse.reorder(layer_matrix)
        eigvec = sparse_power_iteration(layer_matrix)  # returns [n,] or [n,1]
        return eigvec

    eigenvectors = tf.map_fn(map_fn_body, layers, fn_output_signature=tf.TensorSpec([None], tf.float32))
    return eigenvectors, int_to_layers

def uniform_eigenvector_centrality(self):
    """
    Compute uniform eigenvector centrality across all layers.

    Aggregates adjacency matrices from all layers into a single matrix by summing them.
    Then computes eigenvector centrality on the aggregated representation.

    Returns:
        tf.Tensor: A vector of shape [n_nodes] representing node centralities.
    """

    layers = self.get_layers()

    sum_tensor = None

    for li in layers:
        # Get the sparse tensor for the current layer
        layer_matrix = __get_sparse_tensor_between_layers__(self, [li], [li])
        layer_matrix = tf.sparse.reorder(layer_matrix)  # ensure canonical order
        
        if sum_tensor is None:
            sum_tensor = layer_matrix
        else:
            sum_tensor = tf.sparse.add(sum_tensor, layer_matrix)


    # Compute the eigenvector centrality for the aggregated layer matrix
    eigvec = sparse_power_iteration(sum_tensor)
    return eigvec

def local_heterogeneus_eigenvector_centrality(self: MultilayerNetwork, weights=None):
    """
    Compute local heterogeneous eigenvector centrality.

    Each layer’s adjacency matrix is weighted and combined using a diagonal-weighted version
    of the full multilayer adjacency tensor. Centrality is computed per-layer from the combined structure.

    Args:
        weights (tf.Tensor, optional): A square tensor of shape [n_layers, n_layers] representing
            inter-layer influence weights. Defaults to identity if None.

    Returns:
        Tuple[tf.Tensor, dict, dict, dict, dict]: 
            - Tensor of shape [n_layers, n_nodes] with centralities.
            - Mapping from node names to indices.
            - Mapping from layer names to indices.
            - Reverse mapping from indices to node names.
            - Reverse mapping from indices to layer names.

    Raises:
        ValueError: If `weights` shape does not match the number of layers.
    """
    #weights must be a [n_layers,n_layers] dense tensor, where n_layers is the number of layers in the network
    n_layers = len(self.get_layers())
    tensor, nodes_to_int, layers_to_int, int_to_nodes, int_to_layers = __get_sparse_tensor__(self, return_mappings=True)
    
    if (weights is not None) and (weights.shape[0] != n_layers or weights.shape[1] != n_layers):
        raise ValueError(f"Weights must be a square tensor of shape [{n_layers}, {n_layers}]. Got {weights.shape}.")
    
    if weights is None:
        weights = tf.constant(np.identity(n_layers), dtype=tf.float32)

    A_star = __sparse_masked_tensordot(tensor, weights)

    A_star = tf.sparse.reorder(A_star)  # Ensure the sparse tensor is in canonical order
    
    layers_tensor = tf.convert_to_tensor(list(range(n_layers)), dtype=tf.int32)
    # layers = tf.convert_to_tensor(list(self.get_layers()), dtype=tf.int32)

    def compute_eigvec(li):
        layer_matrix = __slice_sparse_tensor__(A_star, [li,li])
        layer_matrix = tf.sparse.reorder(layer_matrix)
        eigvec = sparse_power_iteration(layer_matrix)
        return eigvec
    
    eigenvectors = tf.map_fn(compute_eigvec, layers_tensor, fn_output_signature=tf.TensorSpec([None], tf.float32))
    return eigenvectors, nodes_to_int, layers_to_int, int_to_nodes, int_to_layers

def global_heterogeneus_eigenvector_centrality(self: MultilayerNetwork, weights=None):
    """
    Compute global heterogeneous eigenvector centrality across all layers.

    Builds a full block adjacency matrix using a Khatri-Rao product weighted by inter-layer influences.
    Centrality is computed over the entire multilayer structure.

    Args:
        weights (tf.Tensor, optional): A square matrix [n_layers, n_layers] representing layer coupling.
            Defaults to identity if None.

    Returns:
        Tuple[tf.Tensor, dict, dict, dict, dict]:
            - Tensor of shape [n_layers, n_nodes] with eigenvector centralities.
            - Node and layer mappings (forward and inverse).

    Raises:
        ValueError: If weights do not match the [n_layers, n_layers] shape.
    """

    
    n_layers = len(self.get_layers())
    
    if (weights is not None) and (weights.shape[0] != n_layers or weights.shape[1] != n_layers):
        raise ValueError(f"Weights must be a square tensor of shape [{n_layers}, {n_layers}]. Got {weights.shape}.")
    
    if weights is None:
        weights = tf.identity(np.identity(n_layers))

    sparse_tensor, nodes_to_int, layers_to_int, int_to_nodes, int_to_layers = __get_sparse_tensor__(self, return_mappings=True)
    
    A_block = khatri_rao_weighted_block_sparse(weights, sparse_tensor)

    A_block = tf.sparse.reorder(A_block)  # Ensure the sparse tensor is in canonical order
    
    eigvec = sparse_power_iteration(A_block)
    #divide eigvec in n_layers parts of size n_nodes
    n_nodes = len(self.slices[0]) 
    eigvec = tf.reshape(eigvec, (n_layers,n_nodes))
    
    return eigvec, nodes_to_int, layers_to_int, int_to_nodes, int_to_layers



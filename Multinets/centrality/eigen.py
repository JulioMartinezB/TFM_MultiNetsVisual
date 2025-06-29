import numpy as np
import tensorflow as tf
from pymnet import MultilayerNetwork
from Multinets.utils_tensorflow import __get_sparse_tensor__, __get_sparse_tensor_between_layers__, __slice_sparse_tensor__



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




def sparse_power_iteration(sparse_matrix, num_iters=1000):
    """
    Perform power iteration on a sparse matrix to compute the dominant eigenvector.

    Args:
        sparse_matrix (tf.SparseTensor): A sparse square matrix.
        num_iters (int): Number of iterations to perform. Default is 1000.

    Returns:
        tf.Tensor: A 1D tensor representing the dominant eigenvector.
    """
    n = sparse_matrix.dense_shape[1]
    sparse_matrix = tf.sparse.reorder(sparse_matrix)  # ensure canonical order
    b_k = tf.random.normal([n, 1])  # column vector

    for _ in range(num_iters):

        b_k1 = tf.sparse.sparse_dense_matmul(sparse_matrix, b_k)
        norm = tf.norm(b_k1)
        b_k = b_k1 / norm

    return tf.squeeze(b_k)

def __sparse_masked_tensordot(A_sparse: tf.SparseTensor, W: tf.Tensor) -> tf.SparseTensor:
    """
    Perform a masked tensordot over a sparse tensor by contracting on the diagonals of the last two dimensions.

    Args:
        A_sparse (tf.SparseTensor): Sparse tensor of shape [n, n, d, d] representing the multilayer adjacency.
        W (tf.Tensor): Dense matrix of shape [d, d] representing inter-layer weights.

    Returns:
        tf.SparseTensor: A masked sparse tensor [n, n, d, d] where non-zero entries exist only at [:,:,k,k].
    """

    A_sparse = tf.sparse.reorder(A_sparse)
    indices = A_sparse.indices     # [nnz, 4]
    values = A_sparse.values       # [nnz]
    dense_shape = A_sparse.dense_shape
    n, _, d1, d2 = dense_shape.numpy()
    assert d1 == d2, "Expected square last dimensions for diagonals."

    # Extract only diagonal entries: where indices[:,2] == indices[:,3]
    is_diag = tf.equal(indices[:, 2], indices[:, 3])
    diag_indices = tf.boolean_mask(indices, is_diag)
    diag_values = tf.boolean_mask(values, is_diag)
    diag_t = diag_indices[:, 2]  # the shared diagonal index t

    output_entries = {}  # (i, j, k) -> value

    for k in range(d1):
        # Select entries with t >= k
        mask_k = tf.greater_equal(diag_t, k)
        selected_indices = tf.boolean_mask(diag_indices, mask_k)
        selected_values = tf.boolean_mask(diag_values, mask_k)
        selected_t = tf.boolean_mask(diag_t, mask_k)

        # Apply weights: W[k, t]
        weights = tf.gather(W[k], selected_t)
        weighted_vals = selected_values * weights

        for idx, val in zip(selected_indices.numpy(), weighted_vals.numpy()):
            i, j, t, _ = idx
            key = (i, j, k, k)
            output_entries[key] = output_entries.get(key, 0.0) + val

    final_indices = tf.constant(list(output_entries.keys()), dtype=tf.int64)
    final_values = tf.constant(list(output_entries.values()), dtype=tf.float32)

    return tf.SparseTensor(indices=final_indices, values=final_values, dense_shape=dense_shape)

def khatri_rao_weighted_block_sparse(W: tf.Tensor, A_sparse: tf.SparseTensor) -> tf.SparseTensor:
    """
    Construct a Khatri-Rao-style block sparse matrix from a sparse multilayer adjacency tensor and a weight matrix.

    Each [i,j] block in the final matrix corresponds to W[i,j] * A_j, where A_j = A[:,:,j,j].

    Args:
        W (tf.Tensor): A weight matrix of shape [m, m], where m is the number of layers.
        A_sparse (tf.SparseTensor): Sparse tensor of shape [n, n, m, m], typically with non-zero entries on [:,:,j,j].

    Returns:
        tf.SparseTensor: A block matrix of shape [n*m, n*m] where each block corresponds to a layer-layer interaction.
    """

    n = tf.cast(A_sparse.dense_shape[0], tf.int32)
    m = tf.shape(W)[0]
    
    indices = A_sparse.indices
    values = A_sparse.values

    # Only keep diagonal slices: i == j in the last two dims
    diag_mask = tf.equal(indices[:, 2], indices[:, 3])
    diag_indices = tf.boolean_mask(indices, diag_mask)
    diag_values = tf.boolean_mask(values, diag_mask)

    output_indices = []
    output_values = []

    for i in range(m):
        for j in range(m):
            # Extract entries of A_j = A[:, :, j, j]
            mask_j = tf.equal(diag_indices[:, 2], j)
            indices_j = tf.boolean_mask(diag_indices, mask_j)[:, :2]  # [row, col]
            values_j = tf.boolean_mask(diag_values, mask_j)

            # Compute block offset
            row_offset = i * n
            col_offset = j * n
            
            # Shift indices into block [i,j]
            shifted_indices = indices_j + tf.cast(tf.stack([row_offset, col_offset]), tf.int64)

            # Scale values by W[i, j]
            scaled_values = tf.cast(W[i, j], tf.float32) * values_j

            output_indices.append(shifted_indices)
            output_values.append(scaled_values)

    all_indices = tf.concat(output_indices, axis=0)
    all_values = tf.concat(output_values, axis=0)
    dense_shape = tf.constant(np.array([n*m, n*m]), dtype=tf.int64)

    return tf.SparseTensor(indices=all_indices, values=all_values, dense_shape=dense_shape)

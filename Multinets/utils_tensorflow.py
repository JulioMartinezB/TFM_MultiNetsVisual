from pymnet import MultilayerNetwork
import tensorflow as tf


def __init_from_sparse_tensor__(sparse_tensor,  nodes_names=None, aspects_names=None,directed=False):
    """
    Initialize a MultilayerNetwork from a TensorFlow SparseTensor.

    Args:
        sparse_tensor (tf.SparseTensor): The input sparse tensor representing a multilayer network.
            Expected shape: (a1, a2, ..., an1, an2, n1, n2), where the first 2k dimensions represent k aspects
            (each with equal size per pair), and the last two dimensions represent nodes.
        nodes_names (list of str, optional): List of node names. If None, nodes are named automatically.
        aspects_names (list of list of str, optional): Layer names for each aspect.
            If None, names are generated as l<aspect>_<layer>.
        directed (bool, optional): Whether the network should be directed. Default is False.

    Returns:
        pymnet.MultilayerNetwork: The resulting multilayer network object.
    
    Raises:
        ValueError: If the input tensor does not follow expected shape constraints.
    """

    if sparse_tensor.shape.rank < 2:
        raise ValueError("The sparse tensor must have at least 2 dimensions.")
    
    if len(sparse_tensor.shape[:-2]) % 2 != 0:
        raise ValueError("The sparse tensor must have an even number of dimensions.")
        
    if any([sparse_tensor.shape[i] != sparse_tensor.shape[i+1] for i in range(0, sparse_tensor.shape.rank-2, 2)]):
        raise ValueError("The sparse tensor must have the same size in the first and second dimensions for each aspect.")
    
    if nodes_names and len(nodes_names) != sparse_tensor.shape[-2]:
        raise ValueError("The number of nodes must be equal to the size of the last two dimensions in the sparse tensor.")
    
    if nodes_names is None:
        nodes_names = [f"n{i}" for i in range(sparse_tensor.shape[-2])]

    if aspects_names:
        if len(aspects_names) != sparse_tensor.shape[:-2].rank/2:
            raise ValueError("The number of aspects must be equal to the number of non-nodes dimensions in the sparse tensor: ", sparse_tensor.shape[:-2])
    
        for i in range(len(aspects_names)):
            if len(aspects_names[i]) != sparse_tensor.shape[2*i] or len(aspects_names[i]) != sparse_tensor.shape[2*i+1]:
                raise ValueError(f"The number of layers in aspect {i} must be equal to the size of dimension {i} in the sparse tensor.")
    else:
        aspects_names = [ [f'l{i}_{j}' for j in range(sparse_tensor.shape[2*i])] for i in range(len(sparse_tensor.shape[:-2])//2) ]
    

    # Initialize the network
    n = MultilayerNetwork(
        aspects=len(aspects_names),
        noEdge=False,
        directed=directed,
        fullyInterconnected=True,
    )

    for node in nodes_names:
        n.add_node(node)

    for i in range(len(aspects_names)):
        for j in range(len(aspects_names[i])):
            n.add_layer(aspects_names[i][j], i+1)

    for (index, value) in zip(sparse_tensor.indices, sparse_tensor.values):

        aspects_index = index[:-2]
        if aspects_index is not None:
            edge = [nodes_names[index[-2]],  nodes_names[index[-1]]] + [aspects_names[i//2][aspects_index[i]] for i in range(len(aspects_index))]
        else:
            edge = [nodes_names[index[-2]],  nodes_names[index[-1]]]
        print(edge)
        n[tuple(edge)] = value

    return n
    
def __get_sparse_tensor__(self, return_mappings=False):
    """
    Convert a MultilayerNetwork to a TensorFlow SparseTensor representation.

    Args:
        return_mappings (bool): Whether to return node/layer mapping dictionaries along with the tensor.

    Returns:
        tf.SparseTensor: Sparse tensor representing the multilayer network.
        
        If `return_mappings` is True, also returns:
            - nodes (dict): Mapping from node name to index.
            - layers (list of dict): Mappings from layer names to indices per aspect.
            - inv_nodes (dict): Reverse mapping of `nodes`.
            - inv_layers (list of dict): Reverse mappings of `layers`.
    """
    
    if 'sparse_tensor' in self.__dict__ and self.sparse_tensor is not None:
        return self.sparse_tensor
    # definde dictionary with name and position of each node
    nodes = {}
    for i in range(len(self.slices[0])):
        nodes[list(self.slices[0])[i]] = i

        
    layers = [ {list(self.slices[i])[j] : j for j in range(len(self.slices[i]))} for i in range(1, self.aspects+1)]
    
    indices = []
    values = []
    for edge in self.edges:
        edge_nodes = edge[:2]
        edge_aspects = edge[2:-1]
        
        edge_aspects = [layers[i//2][edge_aspects[i]] for i in range(len(edge_aspects))]
        edge_nodes = [nodes[edge_nodes[i]] for i in range(len(edge_nodes))]

        indices.append(edge_nodes + edge_aspects)
        values.append(edge[-1])

        if not self.directed:
            inverted_edge = edge_nodes[::-1] + edge_aspects[::-1]
            indices.append(inverted_edge)
            values.append(edge[-1])
    
        

    indices = tf.constant(indices, dtype=tf.int64)
    values = tf.constant(values, dtype=tf.float32)
    
    # shape = [len(self.slices[i]) for i in range(self.aspects+1)]
    shape = [len(self.slices[i//2]) for i in range(0,(self.aspects+1)*2)]

    
    sparse_tensor = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
    sparse_tensor = tf.sparse.reorder(sparse_tensor)  # Ensure the indices are sorted
     
    if return_mappings:
        inv_nodes = {v: k for k, v in nodes.items()}
        inv_layers = [{v: k for k, v in l.items()} for l in layers]
        return sparse_tensor, nodes, layers, inv_nodes, inv_layers
    
    return sparse_tensor


def __get_sparse_tensor_between_layers__(self,l1,l2):
    """
    Extract a sparse tensor slice representing connections between two layers.

    Args:
        l1 (list of str or int): Identifiers (names or indices) for the first set of layers (one per aspect).
        l2 (list of str or int): Identifiers for the second set of layers (one per aspect).

    Returns:
        tf.SparseTensor: A sparse tensor slice showing interactions between layers `l1` and `l2`.

    Raises:
        ValueError: If inputs are malformed or incompatible.
    """
    #generate a list fix_index = [l1[0], l2[0], l1[1], l2[1], ..., l1[n-1], l2[n-1]]

    
    sparse, nodes, layers, inv_nodes, inv_layers = __get_sparse_tensor__(self, return_mappings=True)
    for i in range(len(l1)):
        if isinstance(l1[i], str):
            l1[i] = layers[0][l1[i]]
        if isinstance(l2[i], str):
            l2[i] = layers[0][l2[i]]

    fixed_indices = [item for pair in zip(l1, l2) for item in pair]

    # fixed_indices = []
    # for i in range(len(l1)):
    #     fixed_indices.append(l1[i])
    #     fixed_indices.append(l2[i])
    

    return __slice_sparse_tensor__(sparse, fixed_indices)

def __slice_sparse_tensor__(sparse_tensor, fixed_indices):
    """
    Slice a 2k-dimensional SparseTensor by fixing the last k dimensions.

    Useful for extracting layer-layer connectivity along a specified configuration of layers.

    Args:
        sparse_tensor (tf.SparseTensor): The original sparse tensor of shape [d1, d2, ..., d_{2k}].
        fixed_indices (list of int): Fixed values for the last k dimensions.

    Returns:
        tf.SparseTensor: A lower-dimensional tensor of shape [d1, ..., d_k] where the last k dims were fixed.
    """
    
    fixed_indices = tf.convert_to_tensor(fixed_indices, dtype=tf.int64)
    num_dims = tf.shape(sparse_tensor.dense_shape)[0]
    k = tf.size(fixed_indices)

    
    # Split indices into first k and last k parts
    first_k = sparse_tensor.indices[:, :num_dims - k]
    last_k = sparse_tensor.indices[:, num_dims - k:]
    
    # Create mask for matching fixed_indices (login and between last_k and fixed_indices)
    mask = tf.reduce_all(tf.equal(last_k, fixed_indices), axis=1)

    
    # Filter indices and values
    new_indices = tf.boolean_mask(first_k, mask)
    new_values = tf.boolean_mask(sparse_tensor.values, mask)
    
    # New shape is the first k dimensions
    new_shape = sparse_tensor.dense_shape[:num_dims - k]
    
    return tf.SparseTensor(indices=new_indices, values=new_values, dense_shape=new_shape)




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

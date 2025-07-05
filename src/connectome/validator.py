import numpy as np

def validate_connectome(adj_matrix):
    """Performs basic validation on the connectome adjacency matrix."""
    assert adj_matrix.shape[0] == adj_matrix.shape[1], "Adjacency matrix must be square."
    assert not np.isnan(adj_matrix.data).any(), "Adjacency matrix contains NaNs."
    return True
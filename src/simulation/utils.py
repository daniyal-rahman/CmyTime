import numpy as np
import scipy.sparse as sp


def sparse_mv(mat: sp.spmatrix, vec: np.ndarray) -> np.ndarray:
    """Sparse matrix-vector multiplication returning dense array."""
    return mat.dot(vec)

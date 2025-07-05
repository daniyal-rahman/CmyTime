import numpy as np
import scipy.sparse as sp
from src.simulation.utils import sparse_mv


def test_sparse_mv():
    mat = sp.coo_matrix(np.eye(2))
    vec = np.array([1.0, 2.0])
    out = sparse_mv(mat, vec)
    assert np.allclose(out, vec)

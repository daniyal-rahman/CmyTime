import numpy as np
import scipy.sparse as sp
from pathlib import Path
from src.connectome.exporter import save_sparse_matrix, load_sparse_matrix, save_index_map, load_index_map


def test_save_and_load_matrix(tmp_path):
    m = sp.coo_matrix(np.eye(3))
    path = tmp_path / "mat.npz"
    save_sparse_matrix(path, m)
    loaded = load_sparse_matrix(path)
    assert (loaded != m).nnz == 0


def test_save_and_load_index_map(tmp_path):
    mapping = {"A": 0, "B": 1}
    path = tmp_path / "map.json"
    save_index_map(path, mapping)
    loaded = load_index_map(path)
    assert loaded == mapping

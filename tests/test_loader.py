from pathlib import Path
import scipy.sparse as sp
import json
import numpy as np

PROC = Path("data/processed")

def test_files_exist():
    assert (PROC / "adj_matrices.npz").exists()
    assert (PROC / "neuron_index_map.json").exists()

def test_adjacency_shape_and_symmetry():
    data = np.load(PROC / "adj_matrices.npz", allow_pickle=True)
    chem = sp.csr_matrix(data["chemical"].item())
    elec = sp.csr_matrix(data["electrical"].item())

    assert chem.shape == (279, 279)
    assert elec.shape == (279, 279)
    # Electrical synapses are symmetric
    assert (elec != elec.transpose()).nnz == 0

def test_index_map_consistency():
    with open(PROC / "neuron_index_map.json") as f:
        idx = json.load(f)
    data = np.load(PROC / "adj_matrices.npz", allow_pickle=True)
    chem = sp.csr_matrix(data["chemical"].item())

    assert len(idx) == chem.shape[0]
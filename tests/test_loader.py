import json
import pytest
import scipy.sparse as sp
from pathlib import Path

PROC = Path("data/processed")

def test_files_exist():
    assert (PROC / "adj_chemical.npz").exists()
    assert (PROC / "neuron_index_map.json").exists()

def test_adjacency_shape_and_symmetry():
    chem = sp.load_npz(PROC / "adj_chemical.npz")
    # square
    assert chem.shape[0] == chem.shape[1]
    # no negative weights
    assert chem.data.min() >= 0

def test_index_map_consistency():
    with open(PROC / "neuron_index_map.json") as f:
        idx = json.load(f)
    chem = sp.load_npz(PROC / "adj_chemical.npz")
    # number of neurons
    assert len(idx) == chem.shape[0]
    # every name maps to an integer in range
    for name, i in idx.items():
        assert isinstance(name, str)
        assert isinstance(i, int)
        assert 0 <= i < chem.shape[0]


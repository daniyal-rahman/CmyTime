from pathlib import Path
import json
import scipy.sparse as sp


def save_sparse_matrix(path: Path, matrix: sp.spmatrix) -> None:
    """Save a scipy sparse matrix to NPZ format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, matrix)


def load_sparse_matrix(path: Path) -> sp.spmatrix:
    """Load a scipy sparse matrix saved via ``save_sparse_matrix``."""
    return sp.load_npz(path)


def save_index_map(path: Path, mapping: dict) -> None:
    """Save neuron name -> index mapping as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(mapping, f, indent=2)


def load_index_map(path: Path) -> dict:
    """Load neuron name -> index mapping."""
    with open(path) as f:
        return json.load(f)

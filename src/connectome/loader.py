import os
import json
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp


def load_mat_file(path: Path) -> dict:
    """
    Load a MATLAB .mat file and return its variables dictionary, excluding metadata.
    """
    data = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return {k: v for k, v in data.items() if not k.startswith("__")}


def build_neuron_index_map(neuron_names: np.ndarray) -> dict:
    """
    Given a 1D array of neuron names, build a mapping {name: index}.
    """
    return {str(name): idx for idx, name in enumerate(neuron_names)}


def build_adjacency(matrix: np.ndarray) -> sp.coo_matrix:
    """
    Convert a dense or sparse adjacency array into scipy.sparse COO format.
    """
    if sp.issparse(matrix):
        return matrix.tocoo()
    return sp.coo_matrix(matrix)


def process_connectome(raw_dir: Path, processed_dir: Path):
    """
    Process C. elegans connectome raw .mat files into standardized sparse adjacency and name map.

    Expects raw_dir to contain:
      - ConnOrdered_040903.mat      : adjacency matrix (+ possibly neuron name array)
      - A_sendjoint.mat             : optional gap-junction adjacency
      - NeuronTypeOrdered_040903.mat: secondary file (neuron types, fallback for names)

    Outputs in processed_dir:
      - adj_chemical.npz
      - adj_electrical.npz (if available)
      - neuron_index_map.json
    """
    processed_dir.mkdir(parents=True, exist_ok=True)

    # --- Load adjacency ---
    conn_path = raw_dir / "ConnOrdered_040903.mat"
    conn_data = load_mat_file(conn_path)

    # Find the first square matrix as chemical adjacency
    square = [(k, v) for k, v in conn_data.items()
              if isinstance(v, (np.ndarray, sp.spmatrix)) and getattr(v, 'ndim', None) == 2 and v.shape[0] == v.shape[1]]
    if not square:
        raise KeyError(f"No square adjacency matrix in {conn_path.name}")
    chem_key, chem_mat = square[0]
    chemical = build_adjacency(chem_mat)
    n_neurons = chemical.shape[0]
    sp.save_npz(processed_dir / "adj_chemical.npz", chemical)
    print(f"Loaded chemical adjacency '{chem_key}' with shape {chemical.shape} and {chemical.nnz} synapses.")

    # --- Attempt to extract neuron names from ConnOrdered file ---
    neuron_names = None
    for k, v in conn_data.items():
        if isinstance(v, np.ndarray) and v.shape[0] == n_neurons and v.dtype.kind in ('U', 'S', 'O'):
            try:
                names = np.array([str(x) for x in v.flatten()])
                neuron_names = names
                print(f"Extracted neuron names from conn_data field '{k}' with shape {v.shape}.")
                break
            except Exception:
                continue

    # --- Fallback: load from NeuronTypeOrdered file if above fails ---
    if neuron_names is None:
        type_path = raw_dir / "NeuronTypeOrdered_040903.mat"
        type_data = load_mat_file(type_path)
        for k, v in type_data.items():
            if isinstance(v, np.ndarray) and v.shape[0] == n_neurons and v.dtype.kind in ('U', 'S', 'O'):
                try:
                    neuron_names = np.array([str(x) for x in v.flatten()])
                    print(f"Extracted neuron names from type_data field '{k}' with shape {v.shape}.")
                    break
                except Exception:
                    continue

    if neuron_names is None:
        keys = list(conn_data.keys()) + list(type_data.keys())
        raise KeyError(f"Failed to find neuron names of length {n_neurons}. Available keys: {keys}")

    # --- Save neuron index map ---
    index_map = build_neuron_index_map(neuron_names)
    with open(processed_dir / "neuron_index_map.json", 'w') as f:
        json.dump(index_map, f, indent=2)
    print(f"Saved neuron index map with {len(index_map)} entries.")

    # --- Load electrical adjacency if present ---
    gap_path = raw_dir / "A_sendjoint.mat"
    if gap_path.exists():
        gap_data = load_mat_file(gap_path)
        gap_square = [(k, v) for k, v in gap_data.items()
                      if isinstance(v, (np.ndarray, sp.spmatrix)) and getattr(v, 'ndim', None) == 2 and v.shape == (n_neurons, n_neurons)]
        if gap_square:
            gap_key, gap_mat = gap_square[0]
            electrical = build_adjacency(gap_mat)
            sp.save_npz(processed_dir / "adj_electrical.npz", electrical)
            print(f"Loaded electrical adjacency '{gap_key}' with {electrical.nnz} connections.")
        else:
            print(f"Warning: No matching electrical matrix in {gap_path.name}.")
    else:
        print("No A_sendjoint.mat found; skipping electrical adjacency.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process C. elegans connectome .mat files')
    parser.add_argument('--raw_dir', type=str, default='data/raw/VarshneyEtAl2011')
    parser.add_argument('--processed_dir', type=str, default='data/processed')
    args = parser.parse_args()
    process_connectome(Path(args.raw_dir), Path(args.processed_dir))


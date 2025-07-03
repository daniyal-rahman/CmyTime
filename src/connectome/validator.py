import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def validate_chemical_adjacency(adj: sp.spmatrix):
    # Check square matrix
    rows, cols = adj.shape
    if rows != cols:
        raise ValueError(f"Chemical adjacency is not square: {adj.shape}")
    # Check non-negative weights
    data_min = adj.data.min() if adj.nnz > 0 else 0
    if data_min < 0:
        raise ValueError(f"Chemical adjacency has negative weights: min={data_min}")
    print(f"Chemical adjacency: {rows}x{cols}, {adj.nnz} nonzeros, min weight {data_min}")


def validate_neuron_index_map(index_map: dict, n_neurons: int):
    # Check map length
    if len(index_map) != n_neurons:
        raise ValueError(f"Index map length {len(index_map)} does not match neuron count {n_neurons}")
    # Check indices cover full range
    indices = set(index_map.values())
    expected = set(range(n_neurons))
    missing = expected - indices
    extra = indices - expected
    if missing:
        raise ValueError(f"Missing indices in map: {sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")
    if extra:
        raise ValueError(f"Extra indices in map: {sorted(extra)[:10]}{'...' if len(extra)>10 else ''}")
    print(f"Neuron index map: {n_neurons} entries, full coverage OK")


def validate_electrical_adjacency(adj: sp.spmatrix, n_neurons: int):
    # Check dimensions
    if adj.shape != (n_neurons, n_neurons):
        raise ValueError(f"Electrical adjacency shape {adj.shape} does not match expected {(n_neurons,n_neurons)}")
    # Check symmetry and symmetrize if needed
    diff = adj - adj.T
    if diff.nnz > 0 and np.any(diff.data != 0):
        print("Warning: Electrical adjacency is not symmetric. Symmetrizing by averaging values.")
        # Average the matrix with its transpose
        adj = (adj + adj.T).multiply(0.5)
    print(f"Electrical adjacency: {n_neurons}x{n_neurons}, {adj.nnz} connections, symmetry ensured")
    return adj


def main(processed_dir: Path):
    # Load processed data
    proc = Path(processed_dir)
    chem_path = proc / "adj_chemical.npz"
    map_path = proc / "neuron_index_map.json"
    gap_path = proc / "adj_electrical.npz"

    if not chem_path.exists() or not map_path.exists():
        print("Processed files missing. Run loader first.")
        sys.exit(1)

    # Validate chemical
    chem = sp.load_npz(chem_path)
    with open(map_path) as f:
        index_map = json.load(f)
    n = chem.shape[0]
    validate_chemical_adjacency(chem)
    validate_neuron_index_map(index_map, n)

    # Validate or symmetrize electrical
    if gap_path.exists():
        elec = sp.load_npz(gap_path)
        elec = validate_electrical_adjacency(elec, n)
        # Optionally, save symmetrized electrical adjacency back to file
        sp.save_npz(proc / "adj_electrical.npz", elec)
    else:
        print("No electrical adjacency found; skipping")

    print("All validations passed!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate processed C. elegans connectome files')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Directory containing processed adjacency and index map')
    args = parser.parse_args()
    main(Path(args.processed_dir))


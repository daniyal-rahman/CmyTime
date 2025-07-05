
import numpy as np
import scipy.sparse as sp

def export_to_npz(adj_chemical, adj_electrical, neuron_index_map, path):
    """Exports the connectome data to a compressed NumPy .npz file."""
    sp.save_npz(f"{path}/adj_chemical.npz", adj_chemical)
    sp.save_npz(f"{path}/adj_electrical.npz", adj_electrical)
    np.save(f"{path}/neuron_index_map.npy", neuron_index_map)

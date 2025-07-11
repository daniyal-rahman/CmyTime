
import json
import numpy as np
import scipy.sparse as sp

def load_connectome(path):
    """Loads the C. elegans connectome from processed .npz and .json files."""
    data = np.load(f"{path}/adj_matrices.npz", allow_pickle=True)
    adj_chemical = sp.csr_matrix(data["chemical"].item(), dtype=np.float32)
    adj_electrical = sp.csr_matrix(data["electrical"].item(), dtype=np.float32)

    # Normalize chemical weights to [0, 1]
    if adj_chemical.max() > 0:
        adj_chemical = adj_chemical / adj_chemical.max()

    # Electrical weights are typically binary or represent conductance, normalize to [0, 1] if not already
    if adj_electrical.max() > 0:
        adj_electrical = adj_electrical / adj_electrical.max()

    with open(f"{path}/neuron_index_map.json", 'r') as f:
        neuron_to_idx = json.load(f)

    return adj_chemical, adj_electrical, neuron_to_idx

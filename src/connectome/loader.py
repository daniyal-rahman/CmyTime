
import json
import numpy as np
import scipy.sparse as sp

def load_connectome(path):
    """Loads the C. elegans connectome from processed .npz and .json files."""
    data = np.load(f"{path}/adj_matrices.npz", allow_pickle=True)
    adj_chemical = sp.csr_matrix(data["chemical"].item())
    adj_electrical = sp.csr_matrix(data["electrical"].item())

    with open(f"{path}/neuron_index_map.json", 'r') as f:
        neuron_to_idx = json.load(f)

    return adj_chemical, adj_electrical, neuron_to_idx

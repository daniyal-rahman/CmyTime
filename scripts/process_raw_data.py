
import scipy.io
import numpy as np
import json
import scipy.sparse as sp

def process_raw_data():
    """Processes the raw connectome data from .mat files into a more usable format."""
    conn_ordered = scipy.io.loadmat("data/raw/VarshneyEtAl2011/ConnOrdered_040903.mat")
    neuron_type_ordered = scipy.io.loadmat("data/raw/VarshneyEtAl2011/NeuronTypeOrdered_040903.mat")

    adj_gap_junction = conn_ordered["Ag_t_ordered"].tolil()
    adj_chemical = conn_ordered["A_init_t_ordered"].tocsr()
    neuron_labels = [str(label[0]) for label in conn_ordered["Neuron_ordered"]]

    # Correct superfluous self-loops in the gap junction matrix
    adj_gap_junction[94, 94] = 0
    adj_gap_junction[106, 106] = 0
    adj_gap_junction[216, 216] = 0

    adj_gap_junction = adj_gap_junction.tocsr()

    # Create a dictionary to map neuron labels to indices
    neuron_to_idx = {label: i for i, label in enumerate(neuron_labels)}

    # Save the processed data
    np.savez("data/processed/adj_matrices.npz", chemical=adj_chemical, electrical=adj_gap_junction)
    with open("data/processed/neuron_index_map.json", "w") as f:
        json.dump(neuron_to_idx, f)

if __name__ == "__main__":
    process_raw_data()

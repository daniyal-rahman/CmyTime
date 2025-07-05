import json
import numpy as np
import scipy.sparse as sp

def load_connectome(path):
    """Loads the C. elegans connectome from a JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    neurons = sorted(list(set([c['pre'] for c in data] + [c['post'] for c in data])))
    neuron_to_idx = {neuron: i for i, neuron in enumerate(neurons)}

    n_neurons = len(neurons)
    adj_chemical = sp.lil_matrix((n_neurons, n_neurons))
    adj_electrical = sp.lil_matrix((n_neurons, n_neurons))

    for c in data:
        pre_idx = neuron_to_idx[c['pre']]
        post_idx = neuron_to_idx[c['post']]
        if c['type'] == 'chemical':
            adj_chemical[pre_idx, post_idx] = c['weight']
        elif c['type'] == 'electrical':
            adj_electrical[pre_idx, post_idx] = c['weight']

    return adj_chemical.tocsr(), adj_electrical.tocsr(), neuron_to_idx
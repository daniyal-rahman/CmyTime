import numpy as np
import scipy.sparse as sp
from src.simulation.engine import Engine, NetworkState
from src.models.lif_neuron import LIFNeuron


def test_engine_single_step_spike():
    weights = sp.coo_matrix((1, 1))
    state = NetworkState([LIFNeuron(threshold=0.5)], weights)
    engine = Engine(state)
    spikes = engine.step(np.array([1.0]))
    assert spikes[0] == 1

import numpy as np
from src.models.neuron import Neuron
from src.models.lif_neuron import LIFNeuron


def test_neuron_threshold():
    n = Neuron(threshold=1.0, reset=0.0)
    assert n.step(0.5) == 0
    assert n.step(0.6) == 1
    assert n.v == 0.0


def test_lif_neuron_decay_and_spike():
    n = LIFNeuron(tau=2.0, dt=1.0, threshold=1.0, reset=0.0)
    assert n.step(0.5) == 0
    assert np.isclose(n.v, 0.25)
    assert n.step(1.0) == 1
    assert n.v == 0.0

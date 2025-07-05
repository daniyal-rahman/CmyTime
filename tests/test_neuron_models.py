
from src.models.lif_neuron import LIFNeuron

def test_lif_neuron():
    neuron = LIFNeuron()
    assert neuron.step(0) == 0
    assert neuron.step(100) == 1

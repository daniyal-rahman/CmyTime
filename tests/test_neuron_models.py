from src.models.lif_neuron import LIFNeuron

def test_lif_neuron():
    neuron = LIFNeuron(threshold=-55.0, reset_potential=-70.0)
    # Test no spike
    neuron.membrane_potential = -60.0
    assert neuron.step(0) == 0.0
    assert neuron.membrane_potential < -55.0 # Should decay or stay below threshold

    # Test spike
    neuron.membrane_potential = -50.0 # Above threshold
    assert neuron.step(0) == 1.0
    assert neuron.membrane_potential == neuron.reset_potential # Should reset after spike

    # Test input current causing spike
    neuron = LIFNeuron(tau=1.0, threshold=-55.0, reset_potential=-70.0)
    neuron.membrane_potential = -60.0
    assert neuron.step(100) == 1.0
    assert neuron.membrane_potential == neuron.reset_potential
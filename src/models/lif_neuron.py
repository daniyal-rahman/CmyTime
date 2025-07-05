
from .neuron import Neuron

class LIFNeuron(Neuron):
    def __init__(self, tau=20.0, threshold=-55.0, reset_potential=-70.0):
        super().__init__(initial_potential=reset_potential)
        self.tau = tau
        self.threshold = threshold
        self.reset_potential = reset_potential

    def step(self, input_current):
        self.membrane_potential += (-self.membrane_potential + input_current) / self.tau
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = self.reset_potential
            return 1.0  # Spike
        return 0.0

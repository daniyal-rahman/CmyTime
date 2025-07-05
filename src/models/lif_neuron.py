
from .neuron import Neuron
import numpy as np

class LIFNeuron(Neuron):
    def __init__(self, tau=20.0, threshold=-40.0, reset_potential=-75.0, initial_potential=-50.0):
        super().__init__(initial_potential=initial_potential)
        self.tau = tau
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.last_spike_time = -np.inf # Initialize with a very old spike time

    def step(self, input_current, current_time):
        self.membrane_potential += (-self.membrane_potential + input_current) / self.tau
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = self.reset_potential
            self.last_spike_time = current_time
            # print(f"Neuron spiked at time {current_time}") # Debugging: print when a neuron spikes
            return 1.0  # Spike
        return 0.0

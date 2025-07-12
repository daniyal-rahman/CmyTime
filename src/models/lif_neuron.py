
from .neuron import Neuron
import numpy as np

class LIFNeuron(Neuron):
    def __init__(self, tau=30.0, threshold=-25.0, reset_potential=-50.0, initial_potential=-10.0): # Adjusted for higher excitability
        super().__init__(initial_potential=initial_potential)
        self.tau = tau
        self.threshold = threshold
        self.reset_potential = reset_potential
        self.last_spike_time = -np.inf # Initialize with a very old spike time

    def step(self, input_current, current_time):
        self.membrane_potential += (-self.membrane_potential + input_current) / self.tau
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = self.reset_potential
            self.last_spike_time = current_time + np.random.uniform(-0.1, 0.1)
            # print(f"Neuron {id(self)} spiked at time {current_time}, potential: {self.membrane_potential:.2f}, input: {input_current:.2f}") # Debugging: print when a neuron spikes
            return 1.0  # Spike
        return 0.0


class Neuron:
    def __init__(self, initial_potential=0.0):
        self.membrane_potential = initial_potential

    def step(self, input_current):
        raise NotImplementedError

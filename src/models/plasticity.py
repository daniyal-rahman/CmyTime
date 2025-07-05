import numpy as np
import scipy.sparse as sp

class STDP:
    def __init__(self, learning_rate, tau_plus, tau_minus, a_plus, a_minus):
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus

    def update(self, weights, pre_spikes, post_spikes):
        """Applies STDP updates to the weights (sparse matrix compatible)."""
        # Convert to LIL for efficient modification
        weights_lil = weights.tolil()

        # Get indices of non-zero elements (existing connections)
        rows, cols = weights.nonzero()

        for r, c in zip(rows, cols):
            if pre_spikes[r] and post_spikes[c]:
                # Simplified STDP update for demonstration
                delta_w = self.learning_rate * (self.a_plus * np.exp(-1/self.tau_plus) - self.a_minus * np.exp(-1/self.tau_minus))
                weights_lil[r, c] += delta_w

        return weights_lil.tocsr()
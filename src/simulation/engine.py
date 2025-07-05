from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from ..models.lif_neuron import LIFNeuron
from ..models.plasticity import STDP
from .utils import sparse_mv


@dataclass
class NetworkState:
    neurons: list
    weights: sp.coo_matrix

    def reset(self):
        for n in self.neurons:
            n.reset_state()


@dataclass
class Engine:
    state: NetworkState
    plasticity: STDP | None = None

    def step(self, input_current: np.ndarray) -> np.ndarray:
        # compute synaptic current from spikes of previous step
        spikes = np.array([n.v >= n.threshold for n in self.state.neurons], dtype=float)
        syn_input = sparse_mv(self.state.weights, spikes)
        total_input = syn_input + input_current
        new_spikes = []
        for neuron, inc in zip(self.state.neurons, total_input):
            s = neuron.step(inc)
            new_spikes.append(s)
        new_spikes = np.array(new_spikes, dtype=float)

        if self.plasticity is not None:
            self.state.weights = self.plasticity.update(self.state.weights.tocoo(), spikes, new_spikes)
        return new_spikes

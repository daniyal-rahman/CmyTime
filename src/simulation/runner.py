from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp

from .engine import Engine, NetworkState
from ..models.lif_neuron import LIFNeuron
from ..models.plasticity import STDP, STDPParameters


def build_network(n_neurons: int, weights: sp.spmatrix) -> NetworkState:
    neurons = [LIFNeuron() for _ in range(n_neurons)]
    return NetworkState(neurons=neurons, weights=weights.tocoo())


@dataclass
class Runner:
    engine: Engine

    def run(self, inputs: np.ndarray) -> np.ndarray:
        """Run network over sequence of inputs (T x N)."""
        spikes_over_time = []
        for t_input in inputs:
            spikes = self.engine.step(t_input)
            spikes_over_time.append(spikes)
        return np.stack(spikes_over_time)


def create_default_runner(weights: sp.spmatrix) -> Runner:
    state = build_network(weights.shape[0], weights)
    plasticity = STDP(STDPParameters())
    engine = Engine(state=state, plasticity=plasticity)
    return Runner(engine)

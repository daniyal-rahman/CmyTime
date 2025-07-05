from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp


@dataclass
class STDPParameters:
    a_plus: float = 0.01
    a_minus: float = 0.012
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    w_max: float = 1.0


@dataclass
class STDP:
    params: STDPParameters

    def update(self, weights: sp.coo_matrix, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> sp.coo_matrix:
        """Apply pair-based STDP rule to sparse weights.

        Args:
            weights: Sparse weight matrix (COO).
            pre_spikes: Binary spike vector for presynaptic neurons.
            post_spikes: Binary spike vector for postsynaptic neurons.
        """
        if not sp.isspmatrix_coo(weights):
            weights = weights.tocoo()

        # Compute weight changes for active synapses
        dw_data = []
        for idx, (i, j) in enumerate(zip(weights.row, weights.col)):
            dw = 0.0
            if pre_spikes[i]:
                dw += self.params.a_plus * post_spikes[j]
            if post_spikes[j]:
                dw -= self.params.a_minus * pre_spikes[i]
            dw_data.append(dw)
        new_data = np.clip(weights.data + np.array(dw_data), 0.0, self.params.w_max)
        return sp.coo_matrix((new_data, (weights.row, weights.col)), shape=weights.shape)

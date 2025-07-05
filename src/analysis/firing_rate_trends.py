from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np


def plot_firing_rates(spike_matrix: np.ndarray, path: str | None = None):
    rates = spike_matrix.mean(axis=0)
    plt.figure()
    plt.bar(range(len(rates)), rates)
    plt.xlabel('Neuron')
    plt.ylabel('Mean firing rate')
    if path:
        plt.savefig(path)
    else:
        plt.show()

from __future__ import annotations
import matplotlib.pyplot as plt
import scipy.sparse as sp


def plot_weight_hist(matrix: sp.spmatrix, path: str | None = None):
    data = matrix.data
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel('Weight')
    plt.ylabel('Count')
    if path:
        plt.savefig(path)
    else:
        plt.show()

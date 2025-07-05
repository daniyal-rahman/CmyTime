import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
from src.analysis.visualize_weights import plot_weight_hist
from src.analysis.firing_rate_trends import plot_firing_rates


def test_plot_weight_hist(tmp_path):
    mat = sp.coo_matrix(np.eye(3))
    path = tmp_path / "hist.png"
    plot_weight_hist(mat, path=str(path))
    assert path.exists()


def test_plot_firing_rates(tmp_path):
    spikes = np.zeros((5, 3))
    path = tmp_path / "fr.png"
    plot_firing_rates(spikes, path=str(path))
    assert path.exists()

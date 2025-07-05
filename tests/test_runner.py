import numpy as np
import scipy.sparse as sp
from src.simulation.runner import create_default_runner


def test_runner_output_shape():
    weights = sp.coo_matrix((2, 2))
    runner = create_default_runner(weights)
    inputs = np.zeros((5, 2))
    spikes = runner.run(inputs)
    assert spikes.shape == (5, 2)

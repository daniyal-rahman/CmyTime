import numpy as np
import scipy.sparse as sp
from src.models.plasticity import STDP, STDPParameters


def test_stdp_update():
    params = STDPParameters(a_plus=0.01, a_minus=0.012)
    stdp = STDP(params)
    weights = sp.coo_matrix(([0.5], ([0], [1])), shape=(2, 2))
    pre = np.array([1, 0])
    post = np.array([0, 1])
    new_w = stdp.update(weights, pre, post)
    assert np.isclose(new_w.data[0], 0.498)

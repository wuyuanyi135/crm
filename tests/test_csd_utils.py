import pytest
from crm.utils.csd import create_normal_distribution_n
import numpy as np


def test_create_norm_distribution():
    # test 1d
    ngrid = 50
    count = 1e5
    loc = [50e-6]
    n = create_normal_distribution_n(loc, [10e-6], count, grid_count=ngrid)

    assert n.shape == (ngrid, 2)
    assert np.isclose(n[:, 1].sum(), count, rtol=1e-1)
    assert np.isclose(n[np.argmax(n[:, 1]), 0], loc, rtol=1e-2)

    # test 2d
    ngrid = 50  # each dim
    count = 1e5
    loc = [50e-6, 80e-6]
    scale = [10e-6, 15e-6]
    n = create_normal_distribution_n(loc, scale, count, grid_count=ngrid)

    assert n.shape == (ngrid ** 2, 3)
    assert np.isclose(n[:, 2].sum(), count, rtol=1e-1)
    assert np.allclose(n[np.argmax(n[:, 2]), :-1], loc, rtol=1e-2)
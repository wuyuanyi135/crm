import pytest

from crm.presets.hypothetical import Hypothetical1D
from tests.get_data import get_sample_data
import numpy as np


def test_system_spec_shared():
    # the state should be shallow copied so that the system spec are not copied.
    data = get_sample_data()
    id_ = None
    for d in data:
        if id_ is None:
            id_ = id(d.system_spec)
        else:
            if id_ != id(d.system_spec):
                pytest.fail("system_spec in the result states are different.")


def test_copy_independent():
    spec = Hypothetical1D()
    state = spec.make_empty_state()
    state_copy = state.copy()

    state.temperature += 100
    assert state.temperature != state_copy.temperature

    state.concentration += 100
    assert state.concentration != state_copy.concentration

    state.n.append(np.array([]))
    assert len(state.n) != len(state_copy.n)

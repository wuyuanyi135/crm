import numpy as np
import pytest

from crm.base.state import InletState
from crm.presets.hypothetical import Hypothetical1D
from tests.get_data import get_sample_data


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
    state = spec.make_state()
    state_copy = state.copy()

    state.temperature += 100
    assert state.temperature != state_copy.temperature

    state.concentration += 100
    assert state.concentration != state_copy.concentration

    state.n.append(np.array([]))
    assert len(state.n) != len(state_copy.n)


def test_merge_inlet_states():
    system_spec = Hypothetical1D()
    state1 = system_spec.make_state(state_type=InletState, concentration=1, temperature=25, rt=1)
    state2 = system_spec.make_state(state_type=InletState, concentration=1, temperature=25, rt=0.5)
    merged = state1 + state2
    assert np.isclose(merged.temperature, state1.temperature)
    assert np.isclose(merged.temperature, state2.temperature)

    assert np.isclose(merged.concentration, state1.concentration)
    assert np.isclose(merged.concentration, state2.concentration)

    assert np.isclose(merged.rt, 1 / 3)

    state3 = system_spec.make_state(state_type=InletState, concentration=2, temperature=15, rt=0.5,
                                    n=[np.array([[1e-6, 1e9], [2e-6, 1e8]])])
    merged = state1 + state3
    assert np.isclose(merged.rt, 1 / 3)
    assert np.isclose(merged.temperature, 18.33333333)
    assert np.isclose(merged.concentration, 1.6666666)
    assert np.all(merged.n[0] == state3.n[0])

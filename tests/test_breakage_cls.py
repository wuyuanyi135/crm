import pytest
import numpy as np

from crm.jit.breakage import CubicBreakage, ConstantBreakage
from crm.presets.hypothetical import Hypothetical1D
from crm.utils.csd import create_normal_distribution_n
from tests.test_jit import assert_volume_equal


def test_constant_breakage_class():
    hypo = Hypothetical1D()
    state = hypo.make_state(volume=150e-6, agitation_power=10,
                            n=[create_normal_distribution_n([100e-6], [30e-6], 1e11, )])
    aggl = ConstantBreakage(np.array([(0.5, 3e-5)]))
    B, D = aggl.compute(state, 0)

    form = state.system_spec.forms[0]
    n = state.n[0]
    assert_volume_equal(n, B, D, form)

    assert np.all(D > 0)

    print(f"Solids volume fraction: {form.volume_fraction(n)}")
    print(f"max(D) = {D.max()}")


def test_smoluchowski_break_class():
    hypo = Hypothetical1D()
    hypo.forms[0].supersaturation = lambda s, c: c / s

    state = hypo.make_state(volume=150e-6, agitation_power=0.05, temperature=25,
                            concentration=hypo.forms[0].solubility(t=30),
                            n=[create_normal_distribution_n([100e-6], [30e-6], 1e11, )])
    form = state.system_spec.forms[0]
    aggl = CubicBreakage(np.array([(0.5, 6.25e-5 * 1e18 / 3600)]), -2.15, 1)
    B, D = aggl.compute(state, 0)
    n = state.n[0]
    assert_volume_equal(n, B, D, state.system_spec.forms[0])
    print(D)
    print(f"Solids volume fraction: {form.volume_fraction(n)}")
    print(f"max(D) = {D.max()}")

import pytest
import numpy as np

from crm.base.input import ConstTemperatureInput
from crm.base.output_spec import OutputLastSpec
from crm.mcsolver import MCSolver, MCState, MCSolverOptions
from crm.presets.hypothetical import Hypothetical1D, HypotheticalPolymorphic1D, Hypothetical2D, \
    HypotheticalEqualGrowth2D


def test_simple():
    options = MCSolverOptions(attach_extra=True)

    system_spec = Hypothetical1D()
    concentration = system_spec.forms[0].solubility(60)
    state = system_spec.make_empty_state(concentration=concentration, temperature=25)

    solver = MCSolver(system_spec, options)
    input_ = ConstTemperatureInput(25.)
    state_output = solver.compute(state, 3600, input_)


def test_polymorphic():
    options = MCSolverOptions(attach_extra=True, profiling=True)

    system_spec = HypotheticalPolymorphic1D()
    concentration = system_spec.forms[0].solubility(60)
    state = system_spec.make_empty_state(concentration=concentration, temperature=25)

    solver = MCSolver(system_spec, options)
    input_ = ConstTemperatureInput(25.)
    state_output = solver.compute(state, 3600, input_)

    for state in state_output:
        for i, n in enumerate(state.n):
            if n.shape[0] > 0:
                assert np.all(n[:, 0] >= 0), f"form {i} contains negative dimension"


def test_2d():
    options = MCSolverOptions(attach_extra=True, profiling=True)
    system_spec = Hypothetical2D()
    concentration = system_spec.forms[0].solubility(60)
    state = system_spec.make_empty_state(concentration=concentration, temperature=25)

    solver = MCSolver(system_spec, options)
    input_ = ConstTemperatureInput(25.)
    state_output = solver.compute(state, 3600, input_)

    assert True


def test_equal_growth_2d_should_match_1d():
    options = MCSolverOptions(attach_extra=True, profiling=True, output_spec=OutputLastSpec())
    system_spec_2d = HypotheticalEqualGrowth2D()
    system_spec_1d = Hypothetical1D()
    concentration = system_spec_2d.forms[0].solubility(60)

    state = system_spec_2d.make_empty_state(concentration=concentration, temperature=25)
    solver2d = MCSolver(system_spec_2d, options)
    input_ = ConstTemperatureInput(25.)
    state_output_2d = solver2d.compute(state, 3600, input_)

    state = system_spec_1d.make_empty_state(concentration=concentration, temperature=25)
    solver1d = MCSolver(system_spec_1d, options)
    input_ = ConstTemperatureInput(25.)
    state_output_1d = solver1d.compute(state, 3600, input_)

    assert np.isclose(state_output_1d[-1].concentration, state_output_2d[-1].concentration)

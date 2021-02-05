import pytest
import numpy as np

from crm.base.input import ConstTemperatureInput
from crm.mcsolver import MCSolver, MCState, MCSolverOptions
from crm.presets.hypothetical import Hypothetical1D, HypotheticalPolymorphic1D, Hypothetical2D


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

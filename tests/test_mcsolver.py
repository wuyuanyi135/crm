import pytest

from crm.base.input import ConstTemperatureInput
from crm.mcsolver import MCSolver, MCState, MCSolverOptions
from crm.presets.hypothetical import Hypothetical1D, HypotheticalPolymorphic1D


def test_simple():
    options = MCSolverOptions(attach_extra=True)

    system_spec = Hypothetical1D()
    concentration = system_spec.forms[0].solubility(60)
    state = system_spec.make_empty_state(concentration=concentration, temperature=25)

    solver = MCSolver(system_spec, options)
    input_ = ConstTemperatureInput(25.)
    state_output = solver.compute(state, 300, input_)

def test_polymorphic():
    options = MCSolverOptions(attach_extra=True)

    system_spec = HypotheticalPolymorphic1D()
    concentration = system_spec.forms[0].solubility(60)
    state = system_spec.make_empty_state(concentration=concentration, temperature=25)

    solver = MCSolver(system_spec, options)
    input_ = ConstTemperatureInput(25.)
    state_output = solver.compute(state, 300, input_)

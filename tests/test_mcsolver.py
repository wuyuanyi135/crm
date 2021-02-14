import numpy as np

from crm.base.input import ConstantTemperatureInput, ContinuousInput
from crm.base.output_spec import OutputLastSpec, OutputAllSpec
from crm.base.state import InletState
from crm.solvers.mcsolver import MCSolver, MCSolverOptions
from crm.presets.hypothetical import Hypothetical1D, HypotheticalPolymorphic1D, HypotheticalEqualGrowth2D, HypotheticalPolymorphicEqualGrowth2D


def test_simple():
    options = MCSolverOptions(attach_extra=True)

    system_spec = Hypothetical1D()
    concentration = system_spec.forms[0].solubility(t=60)
    state = system_spec.make_state(concentration=concentration, temperature=25)

    solver = MCSolver(system_spec, options)
    input_ = ConstantTemperatureInput(25.)
    state_output = solver.compute(state, 3600, input_)


def test_polymorphic():
    options = MCSolverOptions(attach_extra=True, profiling=True)

    system_spec = HypotheticalPolymorphic1D()
    concentration = system_spec.forms[0].solubility(t=60)
    state = system_spec.make_state(concentration=concentration, temperature=25)

    solver = MCSolver(system_spec, options)
    input_ = ConstantTemperatureInput(25.)
    state_output = solver.compute(state, 3600, input_)

    for state in state_output:
        for i, n in enumerate(state.n):
            if n.shape[0] > 0:
                assert np.all(n[:, 0] >= 0), f"form {i} contains negative dimension"


def test_equal_growth_2d_should_match_1d():
    options = MCSolverOptions(attach_extra=True, profiling=True, output_spec=OutputLastSpec())
    system_spec_2d = HypotheticalEqualGrowth2D()
    system_spec_1d = Hypothetical1D()
    concentration = system_spec_2d.forms[0].solubility(t=60)

    state = system_spec_2d.make_state(concentration=concentration, temperature=25)
    solver2d = MCSolver(system_spec_2d, options)
    input_ = ConstantTemperatureInput(25.)
    state_output_2d = solver2d.compute(state, 3600, input_)

    state = system_spec_1d.make_state(concentration=concentration, temperature=25)
    solver1d = MCSolver(system_spec_1d, options)
    input_ = ConstantTemperatureInput(25.)
    state_output_1d = solver1d.compute(state, 3600, input_)

    assert np.isclose(state_output_1d[-1].concentration, state_output_2d[-1].concentration)


def test_equal_growth_2d_polymorph_match_1d_polymorph():
    options = MCSolverOptions(attach_extra=True, profiling=True, output_spec=OutputLastSpec())
    system_spec_2d = HypotheticalPolymorphicEqualGrowth2D()
    system_spec_1d = HypotheticalPolymorphic1D()
    concentration = system_spec_2d.forms[0].solubility(t=60)

    state = system_spec_2d.make_state(concentration=concentration, temperature=25)
    solver2d = MCSolver(system_spec_2d, options)
    input_ = ConstantTemperatureInput(25.)
    state_output_2d = solver2d.compute(state, 3600, input_)

    state = system_spec_1d.make_state(concentration=concentration, temperature=25)
    solver1d = MCSolver(system_spec_1d, options)
    input_ = ConstantTemperatureInput(25.)
    state_output_1d = solver1d.compute(state, 3600, input_)

    assert np.isclose(state_output_1d[-1].concentration, state_output_2d[-1].concentration)


def test_continuous():
    options = MCSolverOptions(attach_extra=True, profiling=True, output_spec=OutputLastSpec())
    system_spec = Hypothetical1D()
    concentration = system_spec.forms[0].solubility(t=60)
    state = system_spec.make_state(concentration=concentration, temperature=25)
    solver = MCSolver(system_spec, options)

    inlet_state = system_spec.make_state(state_type=InletState, concentration=concentration, rt=600)
    input_ = ContinuousInput(inlet_state)
    output = solver.compute(state, 3600, input_=input_)
    assert True

def test_starting_from_equilibrium():
    input_ = ConstantTemperatureInput(25.)
    system_spec = Hypothetical1D()
    initial_condition = system_spec.make_state(concentration=system_spec.forms[0].solubility(t=25), temperature=25,
                                               volume=150e-6)
    options = MCSolverOptions(output_spec=OutputAllSpec(), time_step=1.0, )
    solver = MCSolver(system_spec, options)

    solve_time = 60
    result = solver.compute(initial_condition, solve_time, input_)
    assert len(result) == 60
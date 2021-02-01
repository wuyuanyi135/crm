from crm.base.input import ConstTemperatureInput
from crm.mcsolver import MCSolverOptions, MCSolver
from crm.presets.hypothetical import Hypothetical1D, HypotheticalPolymorphic1D


def get_sample_data(options: MCSolverOptions = None, system_spec=None):
    options = options or MCSolverOptions(attach_extra=True, profiling=True, time_step=10)

    system_spec = system_spec or Hypothetical1D()
    concentration = system_spec.forms[0].solubility(60)
    state = system_spec.make_empty_state(concentration=concentration, temperature=25)

    solver = MCSolver(system_spec, options)
    input_ = ConstTemperatureInput(25.)
    state_output = solver.compute(state, 3600, input_)
    return state_output


def get_sample_data_polymorphic(**kwargs):
    return get_sample_data(system_spec=HypotheticalPolymorphic1D(), **kwargs)

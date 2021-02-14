from crm.base.input import ConstantTemperatureInput
from crm.solvers.mcsolver import MCSolverOptions, MCSolver
from crm.presets.hypothetical import Hypothetical1D, HypotheticalPolymorphic1D


def get_sample_data(options: MCSolverOptions = None, system_spec=None, time=3600, time_step=10, input_=None,
                    temperature=25, concentration=None):
    options = options or MCSolverOptions(attach_extra=True, profiling=True, time_step=time_step)

    system_spec = system_spec or Hypothetical1D()
    concentration = concentration or system_spec.forms[0].solubility(t=60)
    state = system_spec.make_state(concentration=concentration, temperature=temperature)

    solver = MCSolver(system_spec, options)
    input_ = input_ or ConstantTemperatureInput(25.)
    state_output = solver.compute(state, time, input_)
    return state_output


def get_sample_data_polymorphic(**kwargs):
    return get_sample_data(system_spec=HypotheticalPolymorphic1D(), **kwargs)

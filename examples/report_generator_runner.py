from crm.base.state import InletState
from crm.utils.report_generator import ReportGenerator, ReportOptions
from solvers.mcsolver import MCSolver
from crm.base.input import ConstantTemperatureInput, LinearTemperatureInput, ContinuousInput
from tests.get_data import get_sample_data
import crm.presets.hypothetical as presets
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system_spec", choices=presets.__all__)
    parser.add_argument("input_spec", choices=["constant", "linear", "continuous"])
    args = parser.parse_args()
    system_spec = getattr(presets, args.system_spec)()

    input_method = args.input_spec
    concentration = system_spec.forms[0].solubility(60)
    simulation_time = 1800
    timestep = 1
    temperature = 25
    if input_method == "constant":
        input_ = ConstantTemperatureInput(25)
    elif input_method == "linear":
        input_ = LinearTemperatureInput(40, 25, 0.5 / 60)
        temperature = 40
    elif input_method == "continuous":
        input_ = ContinuousInput(
            system_spec.make_state(state_type=InletState, temperature=25, concentration=concentration, rt=300))
        simulation_time = 3600
    else:
        raise ValueError()

    meta = MCSolver.get_meta()
    options = ReportOptions(debug=True)
    generator = ReportGenerator(options, meta, system_spec)

    data = get_sample_data(time=simulation_time, time_step=timestep, system_spec=system_spec, input_=input_,
                           temperature=temperature,
                           concentration=concentration)

    generator.generate_report(data)


if __name__ == '__main__':
    main()

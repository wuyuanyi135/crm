from crm.utils.report_generator import ReportGenerator, ReportOptions
from crm.mcsolver import MCSolver
from crm.base.input import ConstTemperatureInput, LinearTemperatureInput
from tests.get_data import get_sample_data, get_sample_data_polymorphic
import crm.presets.hypothetical as presets
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system_spec", choices=presets.__all__)
    parser.add_argument("input_spec", choices=["constant", "linear"])
    args = parser.parse_args()
    system_spec = getattr(presets, args.system_spec)

    if args.input_spec == "constant":
        input_ = ConstTemperatureInput(25)
        temperature = 25
    elif args.input_spec == "linear":
        input_ = LinearTemperatureInput(40, 25, 0.5 / 60)
        temperature = 40

    meta = MCSolver.get_meta()
    options = ReportOptions(debug=True)
    generator = ReportGenerator(options, meta, system_spec())

    data = get_sample_data(time=1800, time_step=1, system_spec=system_spec(), input_=input_, temperature=temperature)

    generator.generate_report(data)


if __name__ == '__main__':
    main()

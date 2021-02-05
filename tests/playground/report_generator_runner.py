from crm.utils.report_generator import ReportGenerator, ReportOptions
from crm.mcsolver import MCSolver
from tests.get_data import get_sample_data, get_sample_data_polymorphic
import crm.presets.hypothetical as presets
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("system_spec", choices=presets.__all__)
    args = parser.parse_args()
    system_spec = getattr(presets, args.system_spec)

    meta = MCSolver.get_meta()
    options = ReportOptions(debug=True)
    generator = ReportGenerator(options, meta, system_spec())

    data = get_sample_data(time=1800, time_step=1, system_spec=system_spec())

    generator.generate_report(data)


if __name__ == '__main__':
    main()

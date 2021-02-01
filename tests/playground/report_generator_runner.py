from crm.utils.report_generator import ReportGenerator, ReportOptions
from crm.mcsolver import MCSolver
from tests.get_data import get_sample_data, get_sample_data_polymorphic
from crm.presets.hypothetical import Hypothetical1D, HypotheticalPolymorphic1D

def main():
    meta = MCSolver.get_meta()
    options = ReportOptions(debug=True)
    generator = ReportGenerator(options, meta, HypotheticalPolymorphic1D())

    data = get_sample_data_polymorphic()

    generator.generate_report(data)


if __name__ == '__main__':
    main()

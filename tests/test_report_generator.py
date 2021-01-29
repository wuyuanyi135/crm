import unittest
from crm.utils.report_generator import ReportGenerator, ReportOptions

class TestReportGenerator(unittest.TestCase):
    def test_report_generator(self):
        options = ReportOptions()
        generator = ReportGenerator(options)
        generator.generate_report(None)


if __name__ == '__main__':
    unittest.main()

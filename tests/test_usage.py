import unittest

from crm.mcsolver import MCSolver, MCState
from crm.presets.hypothetical import Hypothetical1D

class TestUsage(unittest.TestCase):
    def test_mc_usage(self):
        system_spec = Hypothetical1D()
        solver = MCSolver(system_spec)

        MCState()


if __name__ == '__main__':
    unittest.main()

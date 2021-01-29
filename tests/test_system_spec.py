import unittest

import numpy as np

from crm.presets.hypothetical import Hypothetical1D

class TestSystemSpec(unittest.TestCase):
    def test_hypothetical_kinetics(self):
        system_spec = Hypothetical1D()
        forms = system_spec.forms
        self.assertTrue(len(forms), 1)

        form = forms[0]

        solubility = form.solubility(40)
        self.assertAlmostEqual(solubility, 0.0193, 4)

        ss = system_spec.supersaturation(solubility, 0.028914)
        gd = form.growth_rate(40, ss)
        self.assertAlmostEqual(float(gd), 0.05e-6, 9)

        nr = form.nucleation_rate(40, ss, 0.1)

        self.assertAlmostEqual(nr[0], 25000000)
        self.assertTrue(np.isclose(nr[1], 5.3861e+08))

        # dissolution
        ss = system_spec.supersaturation(solubility, 0.009638)
        gd = form.dissolution_rate(40, ss)
        self.assertAlmostEqual(float(gd), 1.1e-6, 9)
if __name__ == '__main__':
    unittest.main()

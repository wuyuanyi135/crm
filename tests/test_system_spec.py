import pytest

import numpy as np

from crm.base.system_spec import SystemSpec
from crm.presets.hypothetical import Hypothetical1D


def test_hypothetical_kinetics():
    system_spec = Hypothetical1D()
    forms = system_spec.forms
    assert len(forms) == 1

    form = forms[0]

    solubility = form.solubility(40)
    assert np.isclose(solubility, 0.019276)

    ss = system_spec.supersaturation(solubility, 0.028914)
    gd = form.growth_rate(40, ss)
    assert np.isclose(gd, 0.05e-6)

    nr = form.nucleation_rate(40, ss, 0.1)

    assert np.isclose(nr[0], 25000000)
    assert np.isclose(nr[1], 5.3861e+08)

    # dissolution
    ss = system_spec.supersaturation(solubility, 0.009638)
    gd = form.dissolution_rate(40, ss)
    assert np.isclose(gd, -1.1e-6)


def test_automatic_assign_class_name():
    name = "test"
    spec = SystemSpec(name)
    assert spec.name == name

    class TestSpec(SystemSpec):
        def __init__(self):
            super().__init__()

    spec = TestSpec()
    assert spec.name == "TestSpec"

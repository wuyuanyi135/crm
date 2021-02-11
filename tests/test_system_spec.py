import pytest

import numpy as np

from crm.utils.csd import create_normal_distribution_n
from crm.base.system_spec import SystemSpec
from crm.presets.hypothetical import Hypothetical1D, Hypothetical2D


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


@pytest.mark.parametrize("system_spec_class", (Hypothetical1D, Hypothetical2D))
def test_volume_average_size(system_spec_class):
    # empty
    system_spec = system_spec_class()
    state = system_spec.make_state()
    sz = system_spec.forms[0].volume_average_size(state.n[0])[np.newaxis, :]
    dimensionality = system_spec.forms[0].dimensionality
    assert np.allclose(sz, [0] * (dimensionality + 1))
    assert sz.shape[0] == 1
    assert sz.shape[1] == dimensionality + 1

    # non_empty
    num_rows = 200
    lengths = np.random.random((num_rows, dimensionality))
    count = np.random.random((num_rows, 1)) * 1e6
    n = np.hstack([lengths, count])

    sz = system_spec.forms[0].volume_average_size(n)[np.newaxis, :]

    assert sz.shape[0] == 1
    assert sz.shape[1] == dimensionality + 1
    assert np.allclose(sz[:, -1], n[:, -1].sum())
    assert np.allclose(system_spec.forms[0].volume_fraction(n), system_spec.forms[0].volume_fraction(sz))
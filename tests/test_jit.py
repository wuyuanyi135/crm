import pytest
import numpy as np
from crm.utils.jit import volume_average_size_jit
from crm.presets.hypothetical import Hypothetical1D

def test_volume_average_size_jit():
    # only test when count is specified

    system_spec = Hypothetical1D()
    form = system_spec.forms[0]

    n_unity = np.array([(1e-6, 1), (1e-6, 1)])

    va_row = volume_average_size_jit(n_unity, form.volume_fraction_powers, form.shape_factor, 1)
    assert np.isclose(form.volume_fraction(va_row), form.volume_fraction(n_unity))
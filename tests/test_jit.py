import pytest
import numpy as np

from crm.base.system_spec import FormSpec
from crm.utils.jit import volume_average_size_jit, binary_agglomeration_jit
from crm.presets.hypothetical import Hypothetical1D, Hypothetical2D


def test_volume_average_size_jit():
    # only test when count is specified

    system_spec = Hypothetical1D()
    form = system_spec.forms[0]

    n_unity = np.array([(1e-6, 1), (1e-6, 1)])

    va_row = volume_average_size_jit(n_unity, form.volume_fraction_powers, form.shape_factor, 1)[np.newaxis, :]
    assert np.isclose(form.volume_fraction(va_row), form.volume_fraction(n_unity))

def assert_volume_equal_after_agglomeration(n, B, D, form: FormSpec):
    n = n.copy()
    n[:, -1] = D
    v1 = form.volume_fraction(n)
    v2 = form.volume_fraction(B)

    assert np.isclose(v1, v2)



def test_agglomeration_jit_1d():
    system_spec = Hypothetical1D()
    form = system_spec.forms[0]

    n = np.array([(1e-6, 10), (2e-6, 10)])

    B, D = binary_agglomeration_jit(n, 1, form.volume_fraction_powers, form.shape_factor)
    assert_volume_equal_after_agglomeration(n, B, D, form)


    n = np.array([(1e-6, 10), (2e-6, 10), (6e-6, 52)])

    B, D = binary_agglomeration_jit(n, 1, form.volume_fraction_powers, form.shape_factor)
    assert_volume_equal_after_agglomeration(n, B, D, form)


    n = np.array([(1e-6, 10), (2e-6, 10), (6e-6, 52), (8e-6, 42)])

    B, D = binary_agglomeration_jit(n, 1, form.volume_fraction_powers, form.shape_factor)
    assert_volume_equal_after_agglomeration(n, B, D, form)

def test_agglomeration_jit_2d():
    system_spec = Hypothetical2D()
    form = system_spec.forms[0]

    n = np.array([(1e-6, 1e-6, 10), (2e-6, 1e-6, 10), (6e-6, 1e-6, 52), (8e-6, 4e-6, 42)])

    B, D = binary_agglomeration_jit(n, 1, form.volume_fraction_powers, form.shape_factor)
    assert_volume_equal_after_agglomeration(n, B, D, form)

@pytest.mark.parametrize("system_spec_class", [Hypothetical1D, Hypothetical2D])
@pytest.mark.parametrize("nrows", [100, 300, 1000])
def test_benchmark_agglomeration(nrows, system_spec_class, benchmark):
    system_spec = system_spec_class()
    form = system_spec.forms[0]
    dim = form.dimensionality

    sizes = np.random.random((nrows, dim)) * nrows * 1e-6
    cnts = np.random.random((nrows, 1)) * 1e8
    n = np.hstack((sizes, cnts))

    B, D = benchmark(binary_agglomeration_jit, n, 1, form.volume_fraction_powers, form.shape_factor)
    assert_volume_equal_after_agglomeration(n, B, D, form)

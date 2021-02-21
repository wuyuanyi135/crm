from multiprocessing.pool import ThreadPool

import pytest
import numpy as np

from crm.base.system_spec import FormSpec
from crm.jit.agglomeration import binary_agglomeration_jit
from crm.jit.breakage import binary_breakage_jit
from crm.jit.compress import compress_jit
from crm.jit.csd import volume_average_size_jit, volume_fraction_jit, dL_to_dV
from crm.presets.hypothetical import Hypothetical1D, Hypothetical2D
from crm.utils.statistics import weighted_quantile
from crm.utils.csd import create_normal_distribution_n


def test_volume_average_size_jit():
    # only test when count is specified

    system_spec = Hypothetical1D()
    form = system_spec.forms[0]

    n_unity = np.array([(1e-6, 1), (1e-6, 1)])

    va_row = volume_average_size_jit(n_unity, form.volume_fraction_powers, form.shape_factor, mode=-1)[np.newaxis, :]
    assert np.isclose(form.volume_fraction(va_row), form.volume_fraction(n_unity))


def assert_volume_equal(n, B, D, form: FormSpec):
    if B is not None:
        n = n.copy()
        n[:, -1] = D
        v1 = form.volume_fraction(n)
        v2 = form.volume_fraction(B)

        assert np.isclose(v1, v2)
    else:
        assert D is None


def test_agglomeration_jit_1d():
    system_spec = Hypothetical1D()
    form = system_spec.forms[0]

    n = np.array([(1e-6, 10), (2e-6, 10)])
    coef = 2e-14

    B, D = binary_agglomeration_jit(n, form.volume_fraction_powers, form.shape_factor, coef, minimum_count=0)
    assert_volume_equal(n, B, D, form)

    n = np.array([(1e-6, 10), (2e-6, 10), (6e-6, 52)])

    B, D = binary_agglomeration_jit(n, form.volume_fraction_powers, form.shape_factor, coef, minimum_count=0)
    assert_volume_equal(n, B, D, form)

    n = np.array([(1e-6, 10), (2e-6, 10), (6e-6, 52), (8e-6, 42)])

    B, D = binary_agglomeration_jit(n, form.volume_fraction_powers, form.shape_factor, coef, minimum_count=0)
    assert_volume_equal(n, B, D, form)


def test_agglomeration_jit_2d():
    system_spec = Hypothetical2D()
    form = system_spec.forms[0]
    coef = 2e-14
    n = np.array([(1e-6, 1e-6, 10), (2e-6, 1e-6, 10), (6e-6, 1e-6, 52), (8e-6, 4e-6, 42)])

    B, D = binary_agglomeration_jit(n, form.volume_fraction_powers, form.shape_factor, coef, minimum_count=0)
    assert_volume_equal(n, B, D, form)


@pytest.mark.benchmark
@pytest.mark.parametrize("system_spec_class", [Hypothetical1D, Hypothetical2D])
@pytest.mark.parametrize("nrows", [100, 1000])
@pytest.mark.parametrize("compress", [True, False], ids=["compress", "no_compress"])
def test_benchmark_agglomeration(nrows, system_spec_class, compress, benchmark):
    system_spec = system_spec_class()
    form = system_spec.forms[0]
    dim = form.dimensionality
    coef = 2e-14

    loc = 100e-6
    scale = 20e-6
    sizes = np.random.normal(loc=loc, scale=scale, size=(nrows, dim))
    sizes = np.clip(sizes, loc - 2 * scale, loc + 2 * scale)
    cnts = np.random.random((nrows, 1)) * 1e8
    n = np.hstack((sizes, cnts))
    compression_interval = 1e-6 if compress else 0.

    B, D = benchmark(binary_agglomeration_jit, n, form.volume_fraction_powers, form.shape_factor, coef, minimum_count=0,
                     compression_interval=compression_interval)
    assert_volume_equal(n, B, D, form)
    print(f"n rows in B: {B.shape[0]}")


def test_agglomeration_ignore_particles():
    # TODO: another test in real solver to ensure no infinite agglomeration.
    system_spec = Hypothetical1D()
    form = system_spec.forms[0]

    coef = 2e-14
    minimum_count = 1000.

    n = np.array([
        (1e-6, 10),
        (2e-6, 10),
        (3e-6, 10),
    ])

    B, D = binary_agglomeration_jit(n, form.volume_fraction_powers, form.shape_factor, coef,
                                    minimum_count=minimum_count)

    assert B is None and D is None

    n = np.array([
        (1e-6, 10),
        (2e-6, 10),
        (3e-6, 2000),
        (4e-6, 2000),
        (5e-6, 2000),
    ])
    B, D = binary_agglomeration_jit(n, form.volume_fraction_powers, form.shape_factor, coef,
                                    minimum_count=minimum_count)
    assert_volume_equal(n, B, D, form)

    assert B.shape[0] == 3 * 2
    assert D.size == n.shape[0]


@pytest.mark.benchmark
@pytest.mark.parametrize("nrows", [100, 1000, 10000, 100000])
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("scale", [25e-6, 100e-6], ids=["high", "low"])
def test_compression_jit(nrows, ndim, scale, benchmark):
    scale_count = 1e8
    loc = scale * 2
    sizes = np.random.normal(loc=loc, scale=scale, size=(nrows, ndim))
    sizes = np.clip(sizes, loc - scale * 1.5, loc + scale * 1.5)
    count = np.random.random((nrows, 1)) * scale_count
    n = np.hstack((sizes, count))

    if ndim == 1:
        power = np.array([3])
        shape_factor = 0.5
    elif ndim == 2:
        power = np.array([2, 1])
        shape_factor = 0.5
    elif ndim == 3:
        power = np.array([1, 1, 1])
        shape_factor = 0.5

    result = benchmark(compress_jit, n, power, shape_factor)

    original_volume = volume_fraction_jit(n, power, shape_factor)
    result_volume = volume_fraction_jit(result, power, shape_factor)
    assert np.isclose(original_volume, result_volume)
    assert np.isclose(result[:, -1].sum(), n[:, -1].sum())

    for i in reversed(range(ndim)):
        for q in [0.1, 0.5, 0.9]:
            original_q = weighted_quantile(n[:, i], q, n[:, -1])
            result_q = weighted_quantile(result[:, i], q, result[:, -1])
            assert np.isclose(original_q, result_q, atol=8e-6), f"dim {i} q {q} does not match."

    print(f"compressed {nrows} to {result.shape[0]}")


def test_breakage():
    system_spec = Hypothetical1D()
    form = system_spec.forms[0]

    n = np.array([(1e-6, 10)])
    crystallizer_volume = 150e-6  # mL
    kernels = np.array([(0.5, 4.86e15)])
    B, D = binary_breakage_jit(n, kernels, form.volume_fraction_powers, form.shape_factor, crystallizer_volume,
                               minimum_count=0)
    assert_volume_equal(n, B, D, form)

    # test below minimum count
    B, D = binary_breakage_jit(n, kernels, form.volume_fraction_powers, form.shape_factor, crystallizer_volume,
                               minimum_count=1e8)
    if B is not None:
        assert D.size == 1 and D[0] == 0.
        assert B.size == 0


KERNEL_55 = np.array([(0.5, 4.86e15)])
KERNEL_37 = np.array([(0.3, 3e15)])
KERNEL_19 = np.array([(0.1, 1e15)])
KERNEL_LIST = [
    KERNEL_55,
    np.vstack((KERNEL_55, KERNEL_37)),
    np.vstack((KERNEL_55, KERNEL_37, KERNEL_19)),
]


@pytest.mark.benchmark
@pytest.mark.parametrize("kernels", KERNEL_LIST, ids=["kernel_1", "kernel_2", "kernel_3"])
@pytest.mark.parametrize("system_spec_class", [Hypothetical1D, Hypothetical2D])
@pytest.mark.parametrize("nrows", [100, 1000])
@pytest.mark.parametrize("compress", [True, False], ids=["compress", "no_compress"])
def test_benchmark_breakage(kernels, nrows, system_spec_class, compress, benchmark):
    system_spec = system_spec_class()
    form = system_spec.forms[0]
    dim = form.dimensionality

    n = create_normal_distribution_n(loc=[100e-6] * dim, scale=[20e-6] * dim, count_density=1e8, grid_count=nrows)

    compression_interval = 1e-6 if compress else 0.

    # B, D = benchmark(binary_breakage_jit, n, kernels, form.volume_fraction_powers, form.shape_factor,
    #                  crystallizer_volume, compression_interval=compression_interval)
    B, D = binary_breakage_jit(n, kernels, form.volume_fraction_powers, form.shape_factor,
                               compression_interval=compression_interval)
    assert_volume_equal(n, B, D, form)
    print(f"n rows in B: {B.shape[0] if B is not None else 0}")


def test_dL_to_dV():
    size = np.array([2, ])
    dL = 0.5
    shape_factor = 0.6
    vfp = np.array([3, ])
    dV = dL_to_dV(size, dL, vfp, shape_factor)

    assert np.isclose(dV, 3.6)

    # 2D
    size = np.array([2, 4])
    dL = 0.5
    shape_factor = 0.6
    vfp = np.array([1, 2])
    dV = dL_to_dV(size, dL, vfp, shape_factor)
    assert np.isclose(dV, 4.8)

    # 2D
    size = np.array([2, 4])
    dL = 0.5
    shape_factor = 0.6
    vfp = np.array([2, 1])
    dV = dL_to_dV(size, dL, vfp, shape_factor)
    assert np.isclose(dV, 4.8)

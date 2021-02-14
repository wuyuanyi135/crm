import pytest
import numpy as np
from crm.utils.compress import BinningCompressor
from crm.presets.hypothetical import Hypothetical1D, Hypothetical2D
from crm.utils.csd import create_normal_distribution_n
from crm.utils.pandas import StateDataFrame


@pytest.mark.parametrize("system_spec_class", [Hypothetical1D, Hypothetical2D])
def test_jit_consistency(system_spec_class):
    interval = 1e-6
    compressor_nojit = BinningCompressor(grid_interval=interval, jit=False)
    compressor_jit = BinningCompressor(grid_interval=interval, jit=True)

    system_spec = system_spec_class()

    # compress nothing
    state = system_spec.make_state()

    result_nojit = compressor_nojit.compress(state, inplace=False)
    result_jit = compressor_jit.compress(state, inplace=False)

    assert np.allclose(result_nojit.n[0], result_jit.n[0])


@pytest.mark.parametrize("jit", [True, False])
@pytest.mark.parametrize("grid_count", [100, 1000])
@pytest.mark.parametrize("system_spec_class", [Hypothetical1D, Hypothetical2D])
def test_compress(system_spec_class, jit, grid_count, benchmark):
    interval = 1e-6
    compressor = BinningCompressor(grid_interval=interval, jit=jit)

    system_spec = system_spec_class()

    # compress nothing
    state = system_spec.make_state()
    compressor.compress(state)

    assert state.n[0].size == 0

    # compress something
    dimensionality = system_spec.forms[0].dimensionality
    state = system_spec.make_state(
        n=[create_normal_distribution_n([100e-6] * dimensionality, [10e-6] * dimensionality, grid_count=grid_count,
                                        count_density=1e8)])
    new_state = benchmark(compressor.compress, state, inplace=False)

    assert id(new_state) != id(state), "when inplace=False, the state must be copied"

    print(f"Compressed n from {state.n[0].shape[0]} rows to {new_state.n[0].shape[0]}")

    sdf1 = StateDataFrame([state])
    sdf2 = StateDataFrame([new_state])

    assert np.allclose(sdf1.volume_fraction, sdf2.volume_fraction)
    assert np.allclose(sdf1.counts, sdf2.counts)
    assert np.allclose(sdf1.quantiles, sdf2.quantiles, atol=1e-6)

    quantile_rel_change = (sdf2.quantiles - sdf1.quantiles) / sdf1.quantiles
    print(f"Quantile relative change: \n {quantile_rel_change}")

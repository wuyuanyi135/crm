import pytest
import numpy as np
from crm.utils.compress import BinningCompressor
from crm.presets.hypothetical import Hypothetical1D, Hypothetical2D
from crm.utils.csd import create_normal_distribution_n
from crm.utils.pandas import StateDataFrame


@pytest.mark.parametrize("system_spec_class", [Hypothetical1D, Hypothetical2D])
def test_compress(system_spec_class, printer):
    interval = 1e-6
    compressor = BinningCompressor(grid_interval=interval)

    system_spec = system_spec_class()

    # compress nothing
    state = system_spec.make_state()
    compressor.compress(state)

    assert state.n[0].size == 0

    # compress something
    dimensionality = system_spec.forms[0].dimensionality
    state = system_spec.make_state(
        n=[create_normal_distribution_n([100e-6] * dimensionality, [10e-6] * dimensionality, grid_count=200,
                                        count_density=1e8)])
    state_copy = state.copy()
    compressor.compress(state_copy)

    printer(f"Compressed n from {state.n[0].shape[0]} rows to {state_copy.n[0].shape[0]}")

    sdf1 = StateDataFrame([state])
    sdf2 = StateDataFrame([state_copy])

    assert np.allclose(sdf1.volume_fraction, sdf2.volume_fraction)
    assert np.allclose(sdf1.counts, sdf2.counts)
    assert np.allclose(sdf1.quantiles, sdf2.quantiles, rtol=1e-2)

    quantile_rel_change = (sdf2.quantiles - sdf1.quantiles) / sdf1.quantiles
    printer(f"Quantile relative change: \n {quantile_rel_change}")

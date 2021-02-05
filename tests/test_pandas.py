import pytest

from crm.mcsolver import MCSolverOptions
from crm.presets.hypothetical import Hypothetical2D
from tests.get_data import get_sample_data, get_sample_data_polymorphic
from crm.utils.pandas import StateDataFrame
from crm.utils.csd_grid import edges_to_center_grid
import pandas as pd
import numpy as np

sample_data = {
    "1d": get_sample_data(),
    "polymorph_1d": get_sample_data_polymorphic(),
    "2d": get_sample_data(system_spec=Hypothetical2D()),
}


def assert_column_names_match_spec(df, spec):
    assert np.all(df.columns.values == spec.get_form_names())


@pytest.mark.parametrize("sample_data", sample_data.values(), ids=sample_data.keys())
class TestStateDataFrame:
    def test_solubility(self, sample_data):
        sdf_computed = StateDataFrame(sample_data)
        computed_solubility = sdf_computed.solubility

        assert_column_names_match_spec(computed_solubility, sample_data[0].system_spec)

    def test_concentration(self, sample_data):
        sdf = StateDataFrame(sample_data)
        concentration = sdf.concentration
        assert isinstance(concentration, pd.DataFrame)
        assert concentration.columns[0] == "concentration"
        assert concentration.dtypes.iloc[0] == np.float64

    def test_temperature(self, sample_data):
        sdf = StateDataFrame(sample_data)
        temperature = sdf.temperature
        assert isinstance(temperature, pd.DataFrame)
        assert temperature.columns[0] == "temperature"
        assert temperature.dtypes.iloc[0] == np.float64

    def test_supersaturation(self, sample_data):
        # TODO: when solving, the stored computed properties are based on the previous state. When the state is updated,
        #  the stored property could not match the computed property obtained here. This inconsistency should be addressed.
        #  A possible solution is to include the initial state (optionally) in the solution and shift the stored kinetics
        #  so that the computed properties are stored to the state corresponding to them. If use this method, the last
        #  state will be stored without computed properties because they are not useful. When they are used as the initial
        #  condition in the next solution loop, the lost information will be attached to it.
        #  In conclusion, we will remove the redundant extra information attached to the state because either way there will
        #  be some state without correct computed properties assigned.

        sdf = StateDataFrame(sample_data)
        supersaturation_computed = sdf.supersaturation
        assert_column_names_match_spec(supersaturation_computed, sample_data[0].system_spec)

    def test_n(self, sample_data):
        sdf = StateDataFrame(sample_data)
        n = sdf.n
        assert_column_names_match_spec(n, sample_data[0].system_spec)

    def test_counts(self, sample_data):
        sdf = StateDataFrame(sample_data)
        counts_computed = sdf.counts
        assert_column_names_match_spec(counts_computed, sample_data[0].system_spec)

    def test_solid_volume_fraction(self, sample_data):
        sdf = StateDataFrame(sample_data)
        solid_volume_fraction_computed = sdf.volume_fraction
        assert_column_names_match_spec(solid_volume_fraction_computed, sample_data[0].system_spec)

    def test_quantiles(self, sample_data):
        sdf = StateDataFrame(sample_data)
        quantiles = sdf.quantiles
        assert quantiles.columns.nlevels == 3
        assert np.all(quantiles.isna() | quantiles >= 0)

        vol_weighted_quantiles = sdf.volume_weighted_quantiles
        assert vol_weighted_quantiles.columns.nlevels == 3
        assert np.all(vol_weighted_quantiles.isna() | vol_weighted_quantiles >= 0)

    def test_nucleation_rates(self, sample_data):
        sdf = StateDataFrame(sample_data)
        nucleation_rates = sdf.nucleation_rates
        assert nucleation_rates.columns.nlevels == 2

    def test_gds(self, sample_data):
        # TODO: multidimensional is not tested
        sdf = StateDataFrame(sample_data)
        gds = sdf.gds
        assert gds.columns.nlevels == 3

    def test_profiling(self, sample_data):
        sdf = StateDataFrame(sample_data)
        profiling = sdf.profiling
        assert True

    def test_csd(self, sample_data):
        sdf = StateDataFrame(sample_data)
        edges = np.linspace(0, 1000e-6, 100)
        grids = edges_to_center_grid(edges)
        csd = sdf.get_csd(edges)
        lcsd = csd.applymap(lambda x: len(x))
        assert np.all(lcsd == len(grids))

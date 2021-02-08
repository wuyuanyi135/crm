import pytest

from crm.presets.hypothetical import Hypothetical2D, HypotheticalPolymorphicEqualGrowth2D
from tests.get_data import get_sample_data, get_sample_data_polymorphic
from crm.utils.pandas import StateDataFrame
from crm.utils.csd import edges_to_center_grid
import pandas as pd
import numpy as np

sample_data = {
    "1d": get_sample_data(),
    "polymorph_1d": get_sample_data_polymorphic(),
    "2d": get_sample_data(system_spec=Hypothetical2D()),
    "polymorph_2d": get_sample_data(system_spec=HypotheticalPolymorphicEqualGrowth2D()),
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
        csd, edges = sdf.get_csd(edges)
        for csd_each_time in csd.itertuples(False):
            for csd_each_form in csd_each_time:
                assert csd_each_form.shape[0] == len(grids)

    def test_n_rows(self, sample_data):
        sdf = StateDataFrame(sample_data)
        n_rows = sdf.n_rows
        assert_column_names_match_spec(n_rows, sample_data[0].system_spec)

        assert n_rows.iloc[-1, 0] == sdf.n.iloc[-1, 0].shape[0]
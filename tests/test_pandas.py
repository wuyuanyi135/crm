import pytest

from crm.mcsolver import MCSolverOptions
from tests.get_data import get_sample_data, get_sample_data_polymorphic
from crm.utils.pandas import StateDataFrame
import pandas as pd
import numpy as np

sample_data = {
    "sample_state_1d": get_sample_data(),
    "sample_state_polymorph_1d": get_sample_data_polymorphic(),
}


def assert_column_names_match_spec(df, spec):
    assert np.all(df.columns.values == spec.get_form_names())


@pytest.mark.parametrize("sample_data", sample_data.values(), ids=sample_data.keys())
class TestStateDataFrame:
    def test_solubility(self, sample_data):
        sdf = StateDataFrame(sample_data, False)
        solubility = sdf.solubility

        sdf_computed = StateDataFrame(sample_data, True)
        computed_solubility = sdf_computed.solubility

        assert solubility.equals(computed_solubility)

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

        sdf = StateDataFrame(sample_data, use_computed_properties=False)
        supersaturation = sdf.supersaturation
        assert_column_names_match_spec(supersaturation, sample_data[0].system_spec)

        sdf = StateDataFrame(sample_data, use_computed_properties=True)
        supersaturation_computed = sdf.supersaturation
        assert_column_names_match_spec(supersaturation_computed, sample_data[0].system_spec)

        assert supersaturation_computed.equals(supersaturation)

    def test_n(self, sample_data):
        sdf = StateDataFrame(sample_data)
        n = sdf.n
        assert_column_names_match_spec(n, sample_data[0].system_spec)

    def test_counts(self, sample_data):
        sdf = StateDataFrame(sample_data, use_computed_properties=False)
        counts = sdf.counts
        assert_column_names_match_spec(counts, sample_data[0].system_spec)

        sdf = StateDataFrame(sample_data, use_computed_properties=True)
        counts_computed = sdf.counts
        assert_column_names_match_spec(counts_computed, sample_data[0].system_spec)
        assert counts_computed.equals(counts)

    def test_solid_volume_fraction(self, sample_data):
        sdf = StateDataFrame(sample_data, use_computed_properties=False)
        solid_volume_fraction = sdf.solid_volume_fraction
        assert_column_names_match_spec(solid_volume_fraction, sample_data[0].system_spec)

        sdf = StateDataFrame(sample_data, use_computed_properties=True)
        solid_volume_fraction_computed = sdf.solid_volume_fraction
        assert_column_names_match_spec(solid_volume_fraction_computed, sample_data[0].system_spec)

        assert solid_volume_fraction.equals(solid_volume_fraction_computed)

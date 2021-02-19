# Transform the result into easy-to-use pandas dataframes.
import functools
import itertools
from typing import List, Dict

import numpy as np
import pandas as pd

from crm.base.state import State
from crm.base.system_spec import SystemSpec
from crm.utils.statistics import weighted_quantile

TSProperty = pd.DataFrame
# the time series property with row index = time and column = polymorph name.
PolymorphicTSProperty = pd.DataFrame


def series_polymorphic_list_to_dataframe(s: pd.Series):
    return pd.DataFrame(np.array(s.tolist()))


class StateDataFrame:
    """
    Easy to use parsed class of the state list.
    The properties are not mixed in one dataframe because of the difficulty to
    access the mixed single and multi index columns.

    When the object is constructed from the helper function, all fields are
    guaranteed to be filled. All time series will share the same index (time)
    even if they are not available (filled with None)
    """

    @property
    def system_spec(self) -> SystemSpec:
        return self._system_spec

    @property
    def states(self) -> List[State]:
        return self._states

    @property
    def extra(self) -> pd.DataFrame:
        return self.df_extra

    @property
    def time(self) -> pd.Index:
        return self.df_raw.index

    # Bulk properties

    @property
    def concentration(self) -> TSProperty:
        return self.df_raw[["concentration"]]

    @property
    def temperature(self) -> TSProperty:
        return self.df_raw[["temperature"]]

    @functools.cached_property
    def solubility(self) -> PolymorphicTSProperty:
        """
        Solubility of each form
        :return:
        """
        temperature = self.temperature.iloc[:, 0]  # to series
        solubility_per_form = [temperature.apply(lambda x: f.solubility(t=x)).rename(f.name) for f in
                               self.system_spec.forms]
        solubility = pd.concat(solubility_per_form, axis=1)
        return solubility

    @functools.cached_property
    def supersaturation(self) -> PolymorphicTSProperty:
        """
        Supersaturation of each form
        :return:
        """
        solubility = self.solubility
        concentration = self.concentration["concentration"]  # to series

        ss = []
        for s, c in zip(solubility.itertuples(index=False), concentration):
            ss.append([f.supersaturation(s_each_form, c) for s_each_form, f in zip(s, self.system_spec.forms)])
        supersaturation = pd.DataFrame(ss, columns=self.form_names, index=self.time)
        return supersaturation

    # particle scalar properties
    @functools.cached_property
    def counts(self) -> PolymorphicTSProperty:
        """
        Particle counts of each form
        :return:
        """
        n_df = self.n
        counts = n_df.applymap(lambda x: x[:, -1].sum())
        return counts

    @functools.cached_property
    def volume_fraction(self) -> PolymorphicTSProperty:
        """
        volume fraction (volume of solids / volume of liquid) of each form
        :return:
        """
        n_df = self.n
        vf_funcs = [f.volume_fraction for f in self.system_spec.forms]
        vfs = []
        for r in n_df.itertuples(index=False):
            vf = (f(n) for f, n in zip(vf_funcs, r))
            vfs.append(vf)
        vfs_df = pd.DataFrame(vfs, columns=self.form_names, index=self.time)
        return vfs_df

    # particle vector properties
    @functools.cached_property
    def n(self) -> PolymorphicTSProperty:
        """
        unprocessed particle data. Different forms are splitted into multiple columns
        :return:
        """
        n = self.df_raw["n"]  # series
        # cannot use series_polymorphic_list_to_dataframe. It will expand the internal arrays and cause
        # unexpected shape.
        n = pd.DataFrame(n.tolist()).set_axis(self.form_names, axis=1)
        return n

    @functools.cached_property
    def n_rows(self) -> pd.DataFrame:
        """
        Get the number of rows of n at each time point.
        :return:
        """
        n = self.n
        return n.applymap(lambda x: x.shape[0])

    def get_quantiles(self, weight=None) -> PolymorphicTSProperty:
        """
        Multi-index columns:
        level 0 = polymorph names
        level 1 = dimension idx (starting from 0)
        level 2 = quantile names.        :return:
        """
        n = self.n
        quantile_dict = self._quantile_dict
        dimension_ids = [range(f.dimensionality) for f in self._system_spec.forms]
        quantile_names = quantile_dict.keys()

        first_two_levels = []
        for fn, dim in zip(self.form_names, dimension_ids):
            for d in dim:
                first_two_levels.append((fn, d))
        prod = itertools.product(first_two_levels, quantile_names)
        index = pd.MultiIndex.from_tuples([(a, b, c) for (a, b), c in prod], names=["form", "dimension", "quantile"])

        data = []

        for form_name, dimension_id in first_two_levels:
            n_form = n[form_name]  # series, each row is an array
            data_each_form = []
            for _, n_each_row in n_form.iteritems():
                if n_each_row.shape[0] == 0:
                    data_each_form.append(np.ones(len(quantile_dict), ) * np.nan)
                    continue
                sizes = n_each_row[:, dimension_id]
                counts = n_each_row[:, -1]

                if weight == "volume":
                    form = self._system_spec.get_form_by_name(form_name)
                    particle_volumes = form.particle_volume(n_each_row)
                    counts *= particle_volumes
                quantile = weighted_quantile(sizes, list(quantile_dict.values()), sample_weight=counts)
                data_each_form.append(quantile)
            data.append(np.vstack(data_each_form))
        data = np.hstack(data)
        quantile_df = pd.DataFrame(data, columns=index, index=self.time)
        return quantile_df

    @functools.cached_property
    def quantiles(self) -> PolymorphicTSProperty:
        return self.get_quantiles()

    @functools.cached_property
    def volume_weighted_quantiles(self) -> PolymorphicTSProperty:
        return self.get_quantiles(weight="volume")

    def get_csd(self, edge=None, weight=None):
        """
        Get the csd on the given grid. When multidimensional system spec is used, the 1D size distribution of each
        dimension will be returned.
        :param edge:
        :param weight:
        :return:
        level 1: form name
        level 2: dimension name
        Each cell stores a one dimensional np.array.
        """
        n = self.n
        dimension_ids = [range(f.dimensionality) for f in self._system_spec.forms]
        levels = []
        for fn, dim in zip(self.form_names, dimension_ids):
            for d in dim:
                levels.append((fn, d))
        columns = pd.MultiIndex.from_tuples(levels)

        if edge is None:
            # find the maximum range as auto edge
            def find_max(x):
                if x.shape[0] == 0:
                    return 0
                else:
                    return x[:, :-1].max()

            max_vals = n.applymap(find_max)
            edge = np.linspace(0, max_vals.values.max() * 1.1, 100) # TODO: introduce overscale factor

        data = []
        for form_id, form in enumerate(n.columns):
            form_series = n[form]

            def find_hist_each_dim(x, dimensionality):
                ret = []
                for i in range(dimensionality):
                    hist = np.histogram(x[:, i], edge, weights=x[:, -1])[0]
                    ret.append(hist)

                return pd.Series(ret)

            dimensionality = self._system_spec.forms[form_id].dimensionality
            data.append(form_series.apply(lambda x: find_hist_each_dim(x, dimensionality)))
        df = pd.concat(data, axis=1)
        df.columns = columns
        df.index = self.time

        return df, edge

    def get_csd_nd(self, edge_grid, weight=None):
        """
        Return multi-dimensional CSD matching the edge grid.
        :param edge_grid:
        :param weight:
        :return:
        """
        raise NotImplementedError()

    # kinetics:
    @functools.cached_property
    def nucleation_rates(self) -> PolymorphicTSProperty:
        """
        level 0: form name
        level 1: nucleation kinetics name
        :return:
        """
        forms = self._system_spec.forms

        index = pd.MultiIndex.from_product([self.form_names, ("primary", "secondary")],
                                           names=["form", "nucleation_kinetics"])
        data = []
        for s in self.states:
            data.append(np.hstack([f.nucleation_rate(s, i) for i, f in enumerate(forms)]))

        nuc_df = pd.DataFrame(np.vstack(data), columns=index, index=self.time)
        return nuc_df

    @functools.cached_property
    def gds(self) -> PolymorphicTSProperty:
        """
        Growth or dissolution rate
        level 0: form name
        level 1: dimension id
        level 2: evaluation points. When size-independent model is used, it only contains a 0
        :return:
        """
        evaluate_at = self._gds_evaluated_at
        dimension_ids = [range(f.dimensionality) for f in self._system_spec.forms]
        forms = self._system_spec.forms

        first_two_levels = []
        for fn, dim in zip(self.form_names, dimension_ids):
            for d in dim:
                first_two_levels.append((fn, d))
        prod = itertools.product(first_two_levels, evaluate_at.tolist())
        index = pd.MultiIndex.from_tuples([(a, b, c) for (a, b), c in prod], names=["form", "dimension", "evaluate_at"])

        evaluate_at_array = evaluate_at.reshape((-1, 1))
        data = []
        for state in self.states:
            data_each_form = []
            for i, f in enumerate(forms):
                ss = f.state_supersaturation(state, i)
                if ss > f.supersaturation_break_point:
                    gd = f.growth_rate(state, i)
                else:
                    gd = f.dissolution_rate(state, i)
                data_each_form.append(gd)
            data.append(np.hstack(data_each_form))

        gds_df = pd.DataFrame(np.vstack(data), columns=index, index=self.time)
        return gds_df

    # Debugging information
    @functools.cached_property
    def profiling(self) -> pd.DataFrame:
        profiling_dicts = self.df_extra["profiling"].tolist()
        idx = []
        dicts = []
        for i, d in enumerate(profiling_dicts):
            if d is None:
                idx.append(i)
            else:
                dicts.append(d)

        df_profiling = pd.DataFrame(dicts)
        df_profiling.index = self.time[~self.time.isin(idx)]
        return df_profiling

    default_quantiles: Dict[str, float] = {
        "D10": 0.1,
        "D50": 0.5,
        "D90": 0.9
    }

    def __init__(self, states: List[State], quantiles=None, gds_evaluated_at: np.ndarray = None):
        """

        :param states:
        :param quantiles:
        :param gds_evaluated_at: when using size dependent growth, provide the grid to evaluate the size dependent growth
          or dissolution
        """
        self._gds_evaluated_at = gds_evaluated_at or np.array([0])
        if quantiles is None:
            quantiles = self.default_quantiles

        self._quantile_dict = quantiles
        self._states = states

        self._system_spec: SystemSpec = states[0].system_spec
        d = [x.__dict__ for x in states]
        self.df_raw = pd.DataFrame.from_records(d).set_index("time")
        self.df_extra = pd.DataFrame(self.df_raw["extra"].tolist(), index=self.df_raw.index)
        self.form_names = self._system_spec.get_form_names() if self._system_spec is not None else [f"form_{i}" for i in
                                                                                                    range(len(
                                                                                                        self.df_raw[
                                                                                                            "n"].iloc[
                                                                                                            0][0]))]

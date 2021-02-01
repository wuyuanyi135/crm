# Transform the result into easy-to-use pandas dataframes.
import functools
import itertools

from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict

from crm.base.state import State
from crm.utils.statistics import weighted_quantile
import pandas as pd
import numpy as np
from crm.base.system_spec import SystemSpec, FormSpec

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
        solubility_per_form = [temperature.apply(f.solubility).rename(f.name) for f in self.system_spec.forms]
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
            ss.append([self.system_spec.supersaturation(sol, c) for sol in s])
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
        get the csd on the given grid. grid should be the returned value of np.meshgrid
        #TODO: multidimensional is not implemented
        :param edge:
        :param weight:
        :return:
        """
        n = self.n

        return n.applymap(lambda x: np.histogram(x[:, 0], edge, weights=x[:, -1])[0])


    # kinetics:
    @functools.cached_property
    def nucleation_rates(self) -> PolymorphicTSProperty:
        """
        level 0: form name
        level 1: nucleation kinetics name
        :return:
        """
        ss = self.supersaturation
        vf = self.volume_fraction
        t = self.temperature
        forms = self._system_spec.forms

        index = pd.MultiIndex.from_product([self.form_names, ("primary", "secondary")],
                                           names=["form", "nucleation_kinetics"])
        data = []
        for s, v, t in zip(ss.itertuples(index=False), vf.itertuples(index=False), t.itertuples(index=False)):
            data.append(np.hstack([f.nucleation_rate(t[0], s[i], v[i]) for i, f in enumerate(forms)]))

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
        ss = self.supersaturation
        t = self.temperature
        forms = self._system_spec.forms

        first_two_levels = []
        for fn, dim in zip(self.form_names, dimension_ids):
            for d in dim:
                first_two_levels.append((fn, d))
        prod = itertools.product(first_two_levels, evaluate_at.tolist())
        index = pd.MultiIndex.from_tuples([(a, b, c) for (a, b), c in prod], names=["form", "dimension", "evaluate_at"])

        evaluate_at_array = evaluate_at.reshape((-1, 1))
        data = []
        for s, t in zip(ss.itertuples(index=False), t.itertuples(index=False)):
            data_each_form = []
            for f, s_each_form in zip(forms, s):
                if s_each_form > self._system_spec.solubility_break_point:
                    gd = f.growth_rate(t[0], s_each_form, evaluate_at_array)
                else:
                    gd = f.dissolution_rate(t[0], s_each_form, evaluate_at_array)
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

# def make_state_df(
#         states: List[State],
# ) -> StateDataFrame:
#     system_spec = states[0].system_spec
#     # time stamp is the common index
#     d = [x.__dict__ for x in states]
#     df = pd.DataFrame.from_records(d).set_index("time")
#     time = df.index
#     form_names = system_spec.get_form_names() if system_spec is not None else [f"form_{i}" for i in
#                                                                                range(len(df["n"].iloc[0][0]))]
#     df_concentration = df[["concentration"]]
#     df_temperature = df[["temperature"]]
#     df_n = pd.DataFrame(df["n"].tolist(), columns=form_names, index=time)
#
#     df_extra = pd.DataFrame(df["extra"].tolist(), index=time)
#
#     # check whether solubility is available in the extra information
#
#     StateDataFrame(
#         system_spec=system_spec,
#         states=states,
#         time=time,
#         concentration=df_concentration,
#         temperature=df_temperature,
#         n=df_n,
#
#     )
#
#     try:
#         df_extra = pd.DataFrame(df["extra"].tolist(), index=time)
#
#         try:
#             vf_index = pd.MultiIndex.from_product([("vf",), form_names])
#             df_vf = pd.DataFrame(df_extra["vfs"].tolist(), columns=vf_index, index=time)
#             ret["vf"] = df_vf
#         except KeyError:
#             pass
#
#         try:
#             nuc_col_index = pd.MultiIndex.from_product([form_names, ("pn", "sn")])
#             df_nuc = pd.DataFrame(np.array(df_extra["nucleation"].values.tolist()).reshape((-1, 2 * len(form_names))),
#                                   index=time, columns=nuc_col_index).swaplevel(axis=1).sort_index(1)
#             ret["nucleation"] = df_nuc
#         except KeyError:
#             pass
#
#         try:
#             df_profiling = pd.DataFrame.from_dict(df_extra["profiling"].tolist())
#             df_profiling.index = time
#             ret["profiling"] = df_profiling
#         except KeyError:
#             pass
#
#         try:
#             sol_ss_col_index = pd.MultiIndex.from_product([("sol", "ss"), form_names])
#             df_sol_ss = pd.DataFrame(
#                 np.array(df_extra[["sols", "sses"]].values.tolist()).reshape(-1, 2 * len(form_names)),
#                 columns=sol_ss_col_index, index=time)
#             ret["sol"] = df_sol_ss
#         except KeyError:
#             pass
#
#     except KeyError:
#         pass
#
#     return ret

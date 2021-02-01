# Transform the result into easy-to-use pandas dataframes.
import functools
from dataclasses import dataclass
from typing import List, Tuple, Literal, Dict

from crm.base.state import State
import pandas as pd
import numpy as np
from crm.base.system_spec import SystemSpec

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
        if self.use_computed_properties:
            temperature = self.temperature.iloc[:, 0]  # to series
            solubility_per_form = [temperature.apply(f.solubility).rename(f.name) for f in self.system_spec.forms]
            solubility = pd.concat(solubility_per_form, axis=1)
            return solubility
        else:
            solubility = series_polymorphic_list_to_dataframe(self.df_extra["sols"]).set_axis(self.form_names, axis=1)
            solubility.index = self.time
            return solubility

    @functools.cached_property
    def supersaturation(self) -> PolymorphicTSProperty:
        if self.use_computed_properties:
            solubility = self.solubility
            concentration = self.concentration["concentration"]  # to series

            ss = []
            for s, c in zip(solubility.itertuples(index=False), concentration):
                ss.append([self.system_spec.supersaturation(sol, c) for sol in s])
            supersaturation = pd.DataFrame(ss, columns=self.form_names, index=self.time)
            return supersaturation
        else:
            supersaturation = series_polymorphic_list_to_dataframe(self.df_extra["sses"]).set_axis(self.form_names,
                                                                                                   axis=1)
            supersaturation.index = self.time
            return supersaturation

    # particle scalar properties
    @functools.cached_property
    def counts(self) -> PolymorphicTSProperty:
        n_df = self.n
        counts = n_df.applymap(lambda x: x[:, -1].sum())
        return counts

    @functools.cached_property
    def solid_volume_fraction(self) -> PolymorphicTSProperty:
        if self.use_computed_properties:
            n_df = self.n
            vf_funcs = [f.volume_fraction for f in self.system_spec.forms]
            vfs = []
            for r in n_df.itertuples(index=False):
                vf = (f(n) for f, n in zip(vf_funcs, r))
                vfs.append(vf)
            vfs_df = pd.DataFrame(vfs, columns=self.form_names, index=self.time)
            return vfs_df
        else:
            vfs = series_polymorphic_list_to_dataframe(self.df_extra["vfs"]).set_axis(self.form_names, axis=1)
            vfs.index = self.time
            return vfs

    # Multi-index column with polymorph name as the first level and the quantile q value
    # as the second value.
    quantiles: PolymorphicTSProperty

    # particle vector properties
    # unprocessed particle data. Different forms are splitted into multiple columns
    @functools.cached_property
    def n(self) -> PolymorphicTSProperty:
        n = self.df_raw["n"]  # series
        # cannot use series_polymorphic_list_to_dataframe. It will expand the internal arrays and cause
        # unexpected shape.
        n = pd.DataFrame(n.tolist()).set_axis(self.form_names, axis=1)
        return n

    # Multi-index column with polymorph name as the first level and the grid specific size
    # as the second level
    csd: PolymorphicTSProperty
    volume_weighted_csd: PolymorphicTSProperty

    # kinetics:
    # MultiIndex columns with polymorph name as the first level and the type of nucleation
    # (primary, secondary) as the secondary level.
    nucleation: PolymorphicTSProperty

    # MultiIndex columns with polymorph name as the first level and the gd rate on each
    # dimension as the second level. TODO: the third level will be used for size dependent
    # growth that only contains the size of the existing particles in the system
    gds: PolymorphicTSProperty

    # Debugging information
    profiling: pd.DataFrame

    # Unprocessed extra fields. The processed fields will be popped from it.
    extra: pd.DataFrame

    def __init__(self, states: List[State], use_computed_properties: bool = True):
        assert use_computed_properties, "The computed properties are no longer attached to the states."
        self.use_computed_properties = use_computed_properties
        self._states = states

        self._system_spec = states[0].system_spec
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

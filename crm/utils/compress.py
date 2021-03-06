"""
Collection of row reduction algorithms
"""
from typing import List

import numpy as np
from scipy.stats import binned_statistic_dd
import numpy_indexed as npi

from crm.base.state import State
from crm.base.system_spec import FormSpec
from crm.jit.compress import compress_jit


class Compressor:
    """
    Base class for compressors
    The compressor should ensure the balance of count and volume. The size quantiles should be kept as close as possible
    Since the volume average size has been implemented in the system spec, this class's job is to determine how to
    partition the big table so that each partition will be representative and effective in terms of table reduction.
    """

    def compress(self, state: State, inplace=False) -> State:
        pass


class BinningCompressor(Compressor):
    def __init__(self, grid_interval=1e-6, minimum_row=1, jit=False):
        super().__init__()
        self.minimum_row = minimum_row
        self.grid_interval = grid_interval
        self.jit = jit
        if jit:
            self.compress = self.compress_jit

    def compress_jit(self, state: State, inplace=False) -> State:
        if not inplace:
            state = state.copy()

        for i, (n, form) in enumerate(zip(state.n, state.system_spec.forms)):
            if n.size <= self.minimum_row:
                continue

            state.n[i] = compress_jit(n, form.volume_fraction_powers, form.shape_factor, self.grid_interval)
        return state

    def compress(self, state: State, inplace=False) -> State:
        # compute the sample grid
        if not inplace:
            state = state.copy()

        for i, (n, form) in enumerate(zip(state.n, state.system_spec.forms)):
            if n.size <= self.minimum_row:
                continue
            # nbins = []
            # for i_dim in range(n.shape[1] - 1):
            #     nn = n[:, i_dim]
            #     b = np.round((nn.max() - nn.min()) / self.grid_interval + 1)
            #     nbins.append(b)

            # find the binned counts and assignment of each particles to the bins (inverse index)
            # stat, edges, assignments = binned_statistic_dd(n[:, :-1], n[:, -1], statistic="sum", bins=nbins)

            sz = n[:, :-1]
            dims_min = sz.min(axis=0)
            dims_max = sz.max(axis=0)
            interval = self.grid_interval
            cnts = (dims_max - dims_min) // interval + 1
            cumprod_dim = np.cumprod(np.hstack((np.array([1]), cnts)))[:-1]

            assignments = ((sz - dims_min) // interval * cumprod_dim).sum(axis=1)

            # use the index to partition
            partitions = npi.group_by(assignments).split(n)
            equivalent_rows = self.partitions_to_equivalent_rows(partitions, form)
            state.n[i] = equivalent_rows
        return state

    def partitions_to_equivalent_rows(self, partitions: List[np.ndarray], form: FormSpec) -> np.ndarray:
        rows = []
        for p in partitions:
            if p.size == 0:
                continue
            rows.append(form.volume_average_size(p))
        eq_rows = np.vstack(rows)
        return eq_rows

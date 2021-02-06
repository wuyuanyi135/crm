from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Dict, Union, TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from crm.base.system_spec import SystemSpec


@dataclass
class State:
    system_spec: SystemSpec

    time: float = 0

    # solute concentration in kg solute / kg solvent
    concentration: float = 0

    # temperature in degrees c
    temperature: float = 0

    # count density [Colum 1-N: size in meter; Colum N+1 count in #/m3] each element in the list refer to different
    n: List[np.ndarray] = field(default_factory=list)

    # Extra information attached by the solver.
    extra: Union[None, Dict] = None

    def copy(self):
        copied = copy.copy(self)
        copied.n = copy.deepcopy(self.n)
        return copied


@dataclass
class InletState(State):
    """
    States of a inlet stream.
    """
    rt: float = 1

    def __add__(self, other):
        return self.merge_with(other)

    def merge_with(self, other):
        """
        Merge with another inlet state
        :param other:
        :return:
        """
        assert type(other.system_spec) == type(self.system_spec)

        # rt = V/Q so 1/rt is the weight
        rprts = np.array([1 / self.rt, 1 / other.rt])
        total = rprts.sum()
        weights = rprts / total

        # after mixing the rt V/Q1 V/Q2 becomes V/(Q1+Q2)
        rt_new = 1 / total

        temperature = (weights * np.array([self.temperature, other.temperature])).sum()
        concentration = (weights * np.array([self.concentration, other.concentration])).sum()

        n = [np.vstack([self.n[i], other.n[i]]) for i in range(len(self.n))]

        state = InletState(rt=rt_new, temperature=temperature, concentration=concentration, n=n,
                           system_spec=self.system_spec)
        return state


def sample_n_from_distribution(grid: np.ndarray, count: np.ndarray):
    # TODO: multidimensional support
    n = np.vstack([grid, count]).T
    return n


def create_normal_distribution_n(loc, scale, count_density=1, grid_count=50, sigma=2, grid=None, grid_fcn=np.linspace):
    if grid is None:
        grid_low = np.clip(loc - scale * sigma, 0, np.inf)
        grid_high = loc + scale * sigma
        grid = grid_fcn(grid_low, grid_high, grid_count)

    val = norm.pdf(grid, loc=loc, scale=scale) * count_density
    return sample_n_from_distribution(grid, val)

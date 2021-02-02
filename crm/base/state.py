import copy
from dataclasses import dataclass, field
from typing import List, Dict, Union

import numpy as np
from scipy.stats import norm


@dataclass
class State:
    time: float = 0

    # solute concentration in kg solute / kg solvent
    concentration: float = 0

    # temperature in degrees c
    temperature: float = 0

    # count density [Colum 1-N: size in meter; Colum N+1 count in #/m3] each element in the list refer to different
    n: List[np.ndarray] = field(default_factory=list)

    # Extra information attached by the solver.
    extra: Union[None, Dict] = None

    system_spec = None

    def copy(self):
        copied = copy.copy(self)
        copied.n = copy.deepcopy(self.n)
        return copied


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

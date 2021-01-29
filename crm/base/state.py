import copy
from dataclasses import dataclass, field
from typing import List, Dict, Union

import numpy as np


@dataclass
class State:

    # solute concentration in kg solute / kg solvent
    concentration: float = 0

    # temperature in degrees c
    temperature: float = 0

    # count density [Colum 1-N: size in meter; Colum N+1 count in #/m3] each element in the list refer to different
    n: List[np.ndarray] = field(default_factory=list)

    # Extra information attached by the solver.
    extra: Union[None, Dict] = None
    time: float = 0

    def get_count_density(self):
        """
        :return: see the definition of n
        """
        return self.n

    def copy(self):
        return copy.deepcopy(self)

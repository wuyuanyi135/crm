import copy
from dataclasses import dataclass, field
from typing import List, Dict, Union

import numpy as np


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
        copied.n = self.n.copy()
        return copied

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Dict, Union, TYPE_CHECKING

import numpy as np

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

    @staticmethod
    def from_state(state: State, rt: float) -> InletState:
        # Note this state is not copied! the invoker should take care of the reference and copy to the state.
        ret = InletState(rt=rt, **state.__dict__)
        return ret

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



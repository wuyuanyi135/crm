from dataclasses import dataclass, field
from typing import Type, Literal, Optional

import numpy as np

from crm.base.solver import Solver, SolverOptions, SolverMeta
from crm.base.state import State
from crm.base.system_spec import SystemSpec
from crm.utils.compress import Compressor
from crm.utils.csd import edges_to_center_grid


@dataclass
class MCSolverMeta(SolverMeta):
    name: str = "MCSolver"
    version: str = "0.0.1"


@dataclass
class MCState(State):
    """
    State variable for Monte Carlo solver
    """
    n: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MCSolverOptions(SolverOptions):
    time_step: float = 1.0
    nuclei_sizes: np.ndarray = field(default_factory=lambda: np.array([0]))
    time_step_mode: Literal["fixed"] = "fixed"

    compressor: Optional[Compressor] = None


class MCSolver(Solver):
    options: MCSolverOptions

    @staticmethod
    def get_meta() -> SolverMeta:
        return MCSolverMeta()

    def __init__(self, system_spec: SystemSpec, options: MCSolverOptions = None):
        super().__init__(system_spec, options)

    def get_state_type(self) -> Type[State]:
        return MCState

    def get_option_type(self) -> Type[SolverOptions]:
        return MCSolverOptions

    def update_nucleation(self, n: np.ndarray, nucleation_rates: np.ndarray, time_step: float) -> np.ndarray:
        if np.all(nucleation_rates == 0):
            # when no nucleation do not append new row.
            return n
        ncols = n.shape[1]
        nuclei_row = np.zeros((1, ncols))
        nuclei_row[:-1] = self.options.nuclei_sizes
        nuclei_row[0, -1] = time_step * nucleation_rates.sum()
        return np.concatenate([n, nuclei_row], axis=0)

    def update_growth(self, n: np.ndarray, growth_rate: np.ndarray, time_step: float) -> np.ndarray:
        n[:, :-1] += growth_rate.reshape((1, -1)) * time_step
        return n

    def update_dissolution(self, n: np.ndarray, dissolution_rate: np.ndarray, time_step: float) -> np.ndarray:
        n[:, :-1] += dissolution_rate.reshape((1, -1)) * time_step
        positive = np.all(n[:, :-1] >= 0, axis=1)
        n = n[positive, :]
        return n

    def pre_update_n(self, state, **kwargs):
        compressor = self.options.compressor
        if compressor is not None:
            self.make_profiling(kwargs["profiling"], "compress")
            compressor.compress(state, inplace=True)
            self.make_profiling(kwargs["profiling"], "compress")

    def post_solver_step(self, state: State, **kwargs):
        pass

    def get_time_step(self, state: State):
        if self.options.time_step_mode == "fixed":
            return self.options.time_step
        else:
            raise NotImplementedError()

    def update_concentration(self, concentration: float, mass_diffs: np.ndarray, time_step: float) -> float:
        return concentration - mass_diffs.sum() * time_step

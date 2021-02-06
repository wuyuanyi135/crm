import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Type, List, Dict, Tuple

import numpy as np

from crm.base.input import Input, ContinuousInput
from crm.base.output_spec import OutputSpec, OutputAllSpec
from crm.base.state import State, InletState
from crm.base.system_spec import SystemSpec
from crm.utils.git import get_commit_hash
import os, psutil

process = psutil.Process(os.getpid())


@dataclass
class SolverMeta:
    name: str = "solver"
    version: str = "0.0.1"
    commit: str = get_commit_hash()


@dataclass
class SolverOptions:
    output_spec: OutputSpec = field(default_factory=OutputAllSpec)

    # master switch for extra information
    attach_extra: bool = False

    time_step_scale: float = 1.

    max_time_step: float = 10.0

    profiling: bool = False

    # In continuous mode, the removal mechanism does not completely eliminate a blob of particle. This removal ensures
    # finite growth in the long run.
    continuous_leftover_removal_threshold: float = 1e2


class Solver:
    def __init__(self, system_spec: SystemSpec, options: SolverOptions = None):
        self.options = options
        self.system_spec = system_spec

    @staticmethod
    def get_meta() -> SolverMeta:
        raise NotImplementedError()

    def get_option_type(self) -> Type[SolverOptions]:
        return SolverOptions

    def get_state_type(self) -> Type[State]:
        """
        Get the state class associated with this solver
        :return:
        """
        return State

    def get_time_step(self, state: State, growth_or_dissolution: np.ndarray, nucleation_rates: np.ndarray,
                      end_time: float):
        raise NotImplementedError()

    def post_time_step(self, time_steps: np.ndarray):
        options = self.options
        timestep = time_steps.min() * options.time_step_scale
        timestep = options.max_time_step if timestep > options.max_time_step else timestep
        return timestep

    def update_nucleation(self, n: np.ndarray, nucleation_rates: np.ndarray, time_step: float) -> np.ndarray:
        raise NotImplementedError()

    def update_growth(self, n: np.ndarray, growth_rate: np.ndarray, time_step: float) -> np.ndarray:
        raise NotImplementedError()

    def update_dissolution(self, n: np.ndarray, dissolution_rate: np.ndarray, time_step: float) -> np.ndarray:
        raise NotImplementedError()

    def update_concentration(self, concentration: float, mass_diffs: np.ndarray, time_step: float) -> float:
        raise NotImplementedError()

    def update_with_inlet(self, state: State, inlet: InletState, time_step: float) -> State:
        """
        Assume the total volume does not change (i.e., same amount of in and out flows)
        :param state:
        :param inlet:
        :param time_step:
        :return:
        """
        rt = inlet.rt
        state.temperature += time_step / rt * (inlet.temperature - state.temperature)
        state.concentration += time_step / rt * (inlet.concentration - state.concentration)
        n_in = inlet.n.copy()
        for n in n_in:
            n[:, -1] *= time_step / rt

        n_left = state.n.copy()
        for i, n in enumerate(n_left):
            n[:, -1] *= (1 - time_step / rt)
            should_remove = n[:, -1] <= self.options.continuous_leftover_removal_threshold
            # TODO: don't just remove, send them to next stage if available.
            n_left[i] = n[~should_remove]

        for i in range(len(state.n)):
            state.n[i] = np.vstack([n_in[i], n_left[i]])

        return state

    def attach_extra(
            self,
            state: State,
            is_initial: bool = False,
            profiling: dict = None,
    ) -> State:
        options = self.options
        if options.attach_extra:
            extra = {"profiling": profiling, "is_initial": is_initial}
            state.extra = extra
        return state

    def process_output(self, state: State, output_spec: OutputSpec, end_time: float,
                       is_initial: bool = False, **kwargs):
        if output_spec.should_update_output(state, end_time):
            state = state.copy()  # do not update the internal state.
            state = self.attach_extra(state, is_initial, **kwargs)
            output_spec.update_output(state)

    def make_profiling(self, profiling: dict, name: str):
        if profiling is None:
            return

        if name == "ram":
            profiling["ram"] = process.memory_info().rss
            return

        if name in profiling:
            # second entrace
            profiling[name] = time.perf_counter() - profiling[name]
        else:
            profiling[name] = time.perf_counter()

    def compute(
            self,
            init_state: State,
            solve_time: float,
            input_: Input
    ) -> List[State]:
        """
        when t=0, the state equals to the initial state.
        :param init_state:
        :param solve_time:
        :param input_:
        :param options:
        :return:
        """
        options = self.options
        system_spec = self.system_spec
        forms = system_spec.forms
        solubility_break_point = system_spec.solubility_break_point

        state = init_state.copy()
        vfs = np.array([f.volume_fraction(n) for f, n in zip(forms, state.n)])
        densities = np.array([f.density for f in forms])

        end_time = state.time + solve_time

        output_spec = options.output_spec

        # initial condition
        self.process_output(state, output_spec, end_time, is_initial=True)

        while state.time < end_time:
            profiling = {} if options.profiling else None

            # apply input
            state = input_.transform(state)

            # the kinetics of each forms first
            sols = []
            sses = []
            time_steps = []
            nucleation_rate_list = []
            gds = []
            for i, (f, n) in enumerate(zip(forms, state.n)):
                self.make_profiling(profiling, f"kinetics_form_{f.name}")
                sol = f.solubility(state.temperature)
                ss = system_spec.supersaturation(sol, state.concentration)
                if ss == solubility_break_point:
                    nucleation_rates = np.array((0., 0.))
                    gd = 0
                    time_step = np.inf
                elif ss > solubility_break_point:
                    nucleation_rates = f.nucleation_rate(state.temperature, ss, vfs[i])
                    gd = f.growth_rate(state.temperature, ss, n)
                    time_step = self.get_time_step(state, gd, nucleation_rates, end_time)
                else:
                    nucleation_rates = np.array((0., 0.))
                    gd = f.dissolution_rate(state.temperature, ss, n)
                    time_step = self.get_time_step(state, gd, nucleation_rates, end_time)
                sols.append(sol)
                sses.append(ss)
                time_steps.append(time_step)
                nucleation_rate_list.append(nucleation_rates)
                gds.append(gd)
                self.make_profiling(profiling, f"kinetics_form_{f.name}")
            sols = np.array(sols)
            sses = np.array(sses)
            time_steps = np.array(time_steps)

            # use the smaller step size
            time_step = self.post_time_step(time_steps)
            if state.time + time_step > end_time:
                time_step = end_time - state.time

            # if the input contains continuous inlet
            inlet_spec = input_.inlet(state)
            if inlet_spec is not None:
                state = self.update_with_inlet(state, inlet_spec, time_step)
                # re calculate vfs since the CSD may have been modified by the IO flow.
                vfs = np.array([f.volume_fraction(n) for f, n in zip(forms, state.n)])

            # update n
            for i, f in enumerate(forms):
                self.make_profiling(profiling, f"update_n_{f.name}")
                ss = sses[i]
                if ss > solubility_break_point:
                    state.n[i] = self.update_nucleation(state.n[i], nucleation_rate_list[i], time_step)
                    state.n[i] = self.update_growth(state.n[i], gds[i], time_step)
                elif ss < solubility_break_point:
                    state.n[i] = self.update_dissolution(state.n[i], gds[i], time_step)
                self.make_profiling(profiling, f"update_n_{f.name}")
            vfs_new = np.array([f.volume_fraction(n) for f, n in zip(forms, state.n)])

            self.make_profiling(profiling, "update_concentration")
            mass_diffs = (vfs_new - vfs) * densities
            state.concentration = self.update_concentration(state.concentration, mass_diffs, time_step)
            vfs = vfs_new
            self.make_profiling(profiling, "update_concentration")

            state.time += time_step
            self.process_output(state, output_spec, end_time, profiling=profiling)
            self.make_profiling(profiling, "ram")
        return output_spec.get_outputs()

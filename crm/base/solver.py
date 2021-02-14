import os
import time
from dataclasses import dataclass, field
from typing import Type, List

import numpy as np
import psutil

from crm.base.input import Input
from crm.base.output_spec import OutputSpec, OutputAllSpec
from crm.base.state import State, InletState
from crm.base.system_spec import SystemSpec
from crm.utils.git import get_commit_hash

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

    # when continuous input was used, the state might be updated by the inlet condition. This ensures the applied
    # conditions (transform callback) is applied again after the continuous inlet updates the internal states.
    apply_non_continuous_input_twice: bool = True


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

    def get_time_step(self, state: State):
        raise NotImplementedError()

    def post_time_step(self, time_step: float):
        options = self.options
        timestep = time_step * options.time_step_scale
        timestep = options.max_time_step if timestep > options.max_time_step else timestep
        return timestep

    ####################
    # Event callback
    ####################
    def post_apply_continuous_input(self, state, **kwargs):
        pass

    def post_solver_step(self, state: State, **kwargs):
        pass

    def update_nucleation(self, n: np.ndarray, nucleation_rates: np.ndarray, time_step: float) -> np.ndarray:
        raise NotImplementedError()

    def update_growth(self, n: np.ndarray, growth_rate: np.ndarray, time_step: float) -> np.ndarray:
        raise NotImplementedError()

    def update_dissolution(self, n: np.ndarray, dissolution_rate: np.ndarray, time_step: float) -> np.ndarray:
        raise NotImplementedError()

    def update_concentration(self, concentration: float, mass_diffs: np.ndarray, time_step: float) -> float:
        raise NotImplementedError()

    def update_agglomeration(self, n: np.ndarray, B, D, time_step: float) -> np.ndarray:
        """
        Update n with given B and D. If agglomeration is ignored, return the n itself.
        :param n:
        :param B:
        :param D:
        :return:
        """
        return n

    def update_breakage(self, n: np.ndarray, B, D, time_step: float) -> np.ndarray:
        return n

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
        n_in_rate = inlet.n.copy()
        n_in = []
        for n in n_in_rate:
            # must be copied other wise the inlet condition will be accidentally modified.
            n = n.copy()
            n[:, -1] *= time_step / rt
            n_in.append(n)

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
        if output_spec.should_update_output(state, end_time, is_initial=is_initial):
            state = state.copy()  # do not update the internal state.
            state = self.attach_extra(state, is_initial, **kwargs)
            output_spec.update_output(state)

    def make_profiling(self, profiling: dict, name: str, clear=False):
        if profiling is None:
            return

        if name == "ram":
            profiling["ram"] = process.memory_info().rss
            return

        if name in profiling:
            if clear:
                profiling.pop(name)
            else:
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
        :param init_state:
        :param solve_time:
        :param input_:
        :param options:
        :return:
        """
        options = self.options
        system_spec = self.system_spec
        forms = system_spec.forms

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

            # Determine time step with the current state. If the kinetics is required to determine the time step, they
            # will be computed with the system spec bound to the state.
            # This change will potentially harm the performance of batch crystallization because the kinetics may be
            # computed repetitively, but when the state-independent time step is used, or when the system is continuous
            # (the kinetics need to be recalculated again anyway), this might be cleaner
            # TODO throw exception when time step is too large, and return to here to re-calculate the time step.
            time_step = self.get_time_step(state)

            # apply scaling, capping, or custom functions to the time step
            time_step = self.post_time_step(time_step)
            if state.time + time_step > end_time:
                time_step = end_time - state.time

            # if the input contains continuous inlet, update it before the kinetic computation.
            inlet_spec = input_.inlet(state)
            if inlet_spec is not None:
                self.make_profiling(profiling, "upd_cont")
                state = self.update_with_inlet(state, inlet_spec, time_step)

                if options.apply_non_continuous_input_twice:
                    # if the continuous input will change temperature, this option can restore it.
                    state = input_.transform(state)

                # re calculate vfs since the CSD may have been modified by the IO flow.
                vfs = np.array([f.volume_fraction(n) for f, n in zip(forms, state.n)])
                self.make_profiling(profiling, "upd_cont")

                self.post_apply_continuous_input(state, profiling=profiling)

            for i, (f, n) in enumerate(zip(forms, state.n)):
                supersaturation_break_point = f.supersaturation_break_point
                ss = f.state_supersaturation(state, i)

                self.make_profiling(profiling, f"{f.name}_nucgd")
                if ss > supersaturation_break_point:
                    nucleation_rates = f.nucleation_rate(state, i)  # TODO: vfs could be reused
                    state.n[i] = self.update_nucleation(state.n[i], nucleation_rates, time_step)
                    gd = f.growth_rate(state, i)
                    state.n[i] = self.update_growth(state.n[i], gd, time_step)
                elif ss < supersaturation_break_point:
                    gd = f.dissolution_rate(state, i)
                    state.n[i] = self.update_dissolution(state.n[i], gd, time_step)
                # else: do nothing
                self.make_profiling(profiling, f"{f.name}_nucgd")

                self.make_profiling(profiling, f"{f.name}_agg")
                BD = f.agglomeration(state, i)
                if BD is None:
                    self.make_profiling(profiling, f"{f.name}_agg", clear=True)
                else:
                    B, D = BD
                    state.n[i] = self.update_agglomeration(state.n[i], B, D, time_step)
                    self.make_profiling(profiling, f"{f.name}_agg")

                self.make_profiling(profiling, f"{f.name}_brk")
                BD = f.breakage(state, i)
                if BD is None:
                    self.make_profiling(profiling, f"{f.name}_brk", clear=True)
                else:
                    B, D = BD
                    state.n[i] = self.update_breakage(state.n[i], B, D, time_step)
                    self.make_profiling(profiling, f"{f.name}_brk")

            vfs_new = np.array([f.volume_fraction(n) for f, n in zip(forms, state.n)])

            mass_diffs = (vfs_new - vfs) * densities
            state.concentration = self.update_concentration(state.concentration, mass_diffs, time_step)
            vfs = vfs_new

            state.time += time_step

            self.post_solver_step(state, profiling=profiling)

            self.process_output(state, output_spec, end_time, profiling=profiling)
            self.make_profiling(profiling, "ram")

        return output_spec.get_outputs()

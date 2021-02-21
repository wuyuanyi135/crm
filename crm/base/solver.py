import os
import time
from dataclasses import dataclass, field
from typing import Type, List, Tuple, Optional, Sequence

import numpy as np
import psutil

from crm.base.input import Input
from crm.base.output_spec import OutputSpec, OutputAllSpec
from crm.base.state import State, InletState
from crm.base.system_spec import SystemSpec
from crm.utils.compress import Compressor
from crm.utils.git import get_commit_hash

process = psutil.Process(os.getpid())


@dataclass
class SolverMeta:
    name: str = "solver"
    version: str = "0.0.1"
    commit: str = get_commit_hash()


class TimeStepException(Exception):
    def __init__(self, desired_time_step=None):
        super().__init__()
        self.desired_time_step = desired_time_step


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

    def get_time_step(self, state: State, gd, nuc, ss):
        raise NotImplementedError()

    def post_time_step(self, time_step: float):
        options = self.options
        timestep = time_step * options.time_step_scale
        timestep = options.max_time_step if timestep > options.max_time_step else timestep
        return timestep

    def assert_time_step_agg_brk(self, state, time_step, B_agg, D_agg, B_brk, D_brk):
        """
        Verify the time step for agglomeration and breakage is OK. Otherwise, raise TimeStepException
        :param state:
        :param time_step:
        """
        pass

    ####################
    # Event callback
    ####################

    def post_apply_continuous_input(self, state, **kwargs):
        pass

    def post_solver_step(self, state: State, **kwargs):
        pass

    def update_nucleation(self, n: np.ndarray, nucleation_rates: np.ndarray, time_step: float) -> np.ndarray:
        """

        :param n:
        :param nucleation_rates:
        :param time_step:
        :return: updated n with the nucleated row.
        """
        raise NotImplementedError()

    def update_growth(self, n: np.ndarray, growth_rate: np.ndarray, time_step: float) -> np.ndarray:
        raise NotImplementedError()

    def update_dissolution(self, n: np.ndarray, dissolution_rate: np.ndarray, time_step: float) -> np.ndarray:
        """
        After dissolution, the size <= 0 rows should be removed from the table.
        :param n:
        :param dissolution_rate:
        :param time_step:
        :return:
        """
        raise NotImplementedError()

    def update_concentration(self, concentration: float, mass_diffs: np.ndarray, time_step: float) -> float:
        raise NotImplementedError()

    def update_with_D(self, n, time_step, D: Optional[np.ndarray]) -> np.ndarray:
        if D is None:
            return n
        cnt = n[:, -1]
        cnt -= D * time_step
        # assert cnt is not negative.
        n[:, -1] = cnt
        return n

    def update_with_Bs(self, n, time_step, Bs: Sequence[Optional[np.ndarray]]):
        Bs = [B * time_step for B in Bs if B is not None]
        Bs.insert(0, n)
        return np.vstack(Bs)

    def inlet_BD(
            self,
            state: State,
            inlet: InletState,
            time_step: float
    ) -> (List[np.ndarray], List[np.ndarray]):
        """
        Assume the total volume does not change (i.e., same amount of in and out flows)
        :param state: the state will be modified inplace except n, which will be returned as B and D
        :param inlet:
        :param time_step:
        :return: list[B], list[D]
        """
        rt = inlet.rt
        state.temperature += time_step / rt * (inlet.temperature - state.temperature)
        state.concentration += time_step / rt * (inlet.concentration - state.concentration)
        B = []
        for n in inlet.n:
            # must be copied other wise the inlet condition will be accidentally modified.
            n = n.copy()
            n[:, -1] *= time_step / rt
            B.append(n)

        D = []
        for i, n in enumerate(state.n):
            n = n[:, -1].copy()  # D only care about the count
            n *= (time_step / rt)
            D.append(n)

        return B, D

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
        densities = np.array([f.solid_density / f.solvent_density for f in forms])

        end_time = state.time + solve_time

        output_spec = options.output_spec

        # initial condition
        self.process_output(state, output_spec, end_time, is_initial=True)

        # kinetics involved in time step computation
        gd: List[np.ndarray] = [None] * len(forms)
        nuc: List[np.ndarray] = [None] * len(forms)
        ss: List[float] = [None] * len(forms)

        B_cont = None
        D_cont = None

        while state.time < end_time:

            profiling = {} if options.profiling else None

            # apply input
            state = input_.transform(state)

            # compute the kinetics of each form
            for i, (f, n) in enumerate(zip(forms, state.n)):
                supersaturation_break_point = f.supersaturation_break_point
                ss[i] = ss_ = f.state_supersaturation(state, i)

                if ss_ > supersaturation_break_point:
                    # TODO: size dependent growth: gd should include one more row evaluated at the nuclei size.
                    nuc[i] = f.nucleation_rate(state, i, vf=vfs[i])
                    gd[i] = f.growth_rate(state, i)
                elif ss_ < supersaturation_break_point:
                    gd[i] = f.dissolution_rate(state, i)
                # else: do nothing

            time_step = self.get_time_step(state, gd, nuc, ss)
            # apply scaling, capping, or custom functions to the time step
            time_step = self.post_time_step(time_step)
            if state.time + time_step > end_time:
                time_step = end_time - state.time

            # Apply nucleation and growth
            for i, (f, n) in enumerate(zip(forms, state.n)):
                if gd[i] is None:
                    continue

                supersaturation_break_point = f.supersaturation_break_point
                ss_ = ss[i]

                self.make_profiling(profiling, f"upd_{f.name}_nucgd")
                if ss_ > supersaturation_break_point:
                    state.n[i] = self.update_nucleation(state.n[i], nuc[i], time_step)
                    state.n[i] = self.update_growth(state.n[i], gd[i], time_step)
                elif ss_ < supersaturation_break_point:
                    state.n[i] = self.update_dissolution(state.n[i], gd[i], time_step)
                self.make_profiling(profiling, f"upd_{f.name}_nucgd")

            # Apply the concentration update
            vfs_new = np.array([f.volume_fraction(n) for f, n in zip(forms, state.n)])

            mass_diffs = (vfs_new - vfs) * densities
            state.concentration = self.update_concentration(state.concentration, mass_diffs, time_step)
            vfs = vfs_new

            # Apply the continuous IO if any
            inlet_state = input_.inlet(state)
            if inlet_state is not None:
                self.make_profiling(profiling, "upd_cont")

                # the state will be updated (temperature, concentration).
                B_cont, D_cont = self.inlet_BD(state, inlet_state, time_step)
                # TODO: verify inlet state time step here?

                # Apply continuous D only here. B will be applied in the end of the step.
                for i, D in enumerate(D_cont):
                    state.n[i] = self.update_with_D(state.n[i], time_step, D)

                if options.apply_non_continuous_input_twice:
                    # if the continuous input will change temperature, this option can restore it.
                    # No transformation of n is allowed here!
                    state = input_.transform(state)

                # No need to recompute the vfs here since the update on n will not happen until the end of the step.
                self.make_profiling(profiling, "upd_cont")

                self.post_apply_continuous_input(state, profiling=profiling)

            # Compute agg and brk kinetics (BD)
            # The common agg and brk kinetics compute the change rate of number density, not count. To convert the
            # number density to count, we need fit the n into a grid so that the delta L or delta V of the grid can be
            # inserted in the equation.

            B_agg, D_agg, B_brk, D_brk = self.compute_agg_brk_kinetics(state)

            # Verify the time step is ok for agg and brk
            self.assert_time_step_agg_brk(state, time_step, B_agg, D_agg, B_brk, D_brk)

            # Update D of agg and brk
            for i, (D1, D2) in enumerate(zip(D_agg, D_brk)):
                state.n[i] = self.update_with_D(state.n[i], time_step, D1)
                state.n[i] = self.update_with_D(state.n[i], time_step, D2)


            # Update all Bs here
            for i, _ in enumerate(forms):
                Bs = [B_brk[i], B_agg[i]]
                if B_cont is not None:
                    Bs.append(B_cont[i])
                state.n[i] = self.update_with_Bs(state.n[i], time_step, Bs)

            state.time += time_step

            self.post_solver_step(state, profiling=profiling)

            self.process_output(state, output_spec, end_time, profiling=profiling)
            self.make_profiling(profiling, "ram")

        return output_spec.get_outputs()

    def compute_agg_brk_kinetics(self, state, profiling=None):
        """
        Return the BD of agg and brk. Note that this function will also modify the state. E.g., compression before agg.
        :param state:
        :param profiling:
        :return:
        """
        forms = self.system_spec.forms
        B_agg: List[np.ndarray] = [None] * len(forms)
        D_agg: List[np.ndarray] = [None] * len(forms)
        B_brk: List[np.ndarray] = [None] * len(forms)
        D_brk: List[np.ndarray] = [None] * len(forms)

        for i, f in enumerate(forms):
            self.make_profiling(profiling, f"kin_{f.name}_agg")
            B, D = f.agglomeration(state, i)
            if B is not None:
                B_agg[i], D_agg[i] = B, D
            self.make_profiling(profiling, f"kin_{f.name}_agg")

            self.make_profiling(profiling, f"kin_{f.name}_brk")
            B, D = f.breakage(state, i)
            if B is not None:
                B_brk[i], D_brk[i] = B, D
            self.make_profiling(profiling, f"kin_{f.name}_brk")
        return B_agg, D_agg, B_brk, D_brk
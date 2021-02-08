"""
Assemble multiple continuous crystallizer.
"""
from dataclasses import dataclass
from typing import List, Optional

from crm.base.input import Input, ContinuousInput, InputAssembler
from crm.base.output_spec import OutputLastSpec
from crm.base.solver import SolverOptions, Solver
from crm.base.state import State, InletState
from crm.base.system_spec import SystemSpec
from crm.utils.compress import BinningCompressor
from solvers.mcsolver import MCSolver, MCSolverOptions


@dataclass
class StageSpec:
    initial_condition: State

    relative_volume: float = 1.

    # dynamic condition of the stage, not the io flow.
    extra_input: Optional[Input] = None

    # In multistage solution, compressor is required to speed up the process!
    solver_option: SolverOptions = MCSolverOptions(
        output_spec=OutputLastSpec(),
        compressor=BinningCompressor(minimum_row=10000)
    )

    solver_class: Solver = MCSolver


class Sequential:
    def __init__(self, system_spec: SystemSpec, initial_stage_input: ContinuousInput, stage_specs: List[StageSpec],
                 time_step=1, compress_target=100):
        self.compress_target = compress_target
        self.initial_stage_input = initial_stage_input
        self.time_step = time_step
        self.system_spec = system_spec
        self.stage_specs = stage_specs

        self.stage_solvers = []
        self.stage_states = []
        self.stage_extra_inputs = []

        for spec in stage_specs:
            stage_solver = spec.solver_class(system_spec, options=spec.solver_option)
            self.stage_solvers.append(stage_solver)
            self.stage_states.append(spec.initial_condition.copy())
            self.stage_extra_inputs.append(spec.extra_input)

        self.rt_scales = [1]
        first_stage_vol = stage_specs[0].relative_volume
        for spec in stage_specs[1:]:
            self.rt_scales.append(spec.relative_volume / first_stage_vol)

    def compute(self, end_time: float) -> List[List[State]]:
        """

        :return:
        """
        states = self.stage_states
        solvers = self.stage_solvers
        data = []
        while states[0].time < end_time:
            data_each_stage = []
            for i, (solver, ei) in enumerate(zip(solvers, self.stage_extra_inputs)):
                if i == 0:
                    # first stage
                    input_ = self.initial_stage_input
                else:
                    prev_stage_state = InletState.from_state(
                        states[i - 1],
                        self.rt_scales[i] * self.initial_stage_input.inlet_state.rt
                    )
                    # TODO Compressing the inlet state does not help the accumulation of rows in the next stage!
                    # prev_stage_state.n = [compress(n, self.compress_target) for n in prev_stage_state.n]
                    input_ = ContinuousInput(
                        prev_stage_state)

                if ei is not None:
                    input_ = InputAssembler([input_, ei])

                out = solver.compute(states[i], self.time_step, input_)
                s = out[-1]
                states[i] = s
                data_each_stage.append(s)
            data.append(data_each_stage)
        data = [list(x) for x in zip(*data)]  # transpose so that the inner list is the state wrt time
        return data

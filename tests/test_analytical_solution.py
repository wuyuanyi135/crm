import numpy as np
import pytest

from crm.base.input import Input
from crm.base.output_spec import OutputAllSpec, OutputLastSpec
from crm.base.state import State, create_normal_distribution_n
from crm.base.system_spec import SystemSpec, FormSpec
from crm.mcsolver import MCSolver, MCSolverOptions
from crm.utils.pandas import StateDataFrame
from scipy.stats.distributions import norm


def test_pure_growth():
    # define a model that has constant growth rate
    class ConstantGrowthForm(FormSpec):
        def __init__(self, name: str, gds=1e-6):
            super().__init__(name)
            self.gds = gds

        def solubility(self, t: float) -> float:
            return 0.1

        def growth_rate(self, t: float, ss: float, n: np.ndarray = None, state: State = None) -> np.ndarray:
            return np.array(self.gds)

        def dissolution_rate(self, t: float, ss: float, n: np.ndarray = None, state: State = None) -> np.ndarray:
            return np.array(-self.gds)

        def nucleation_rate(self, t: float, ss: float, vf: float, state: State = None) -> np.ndarray:
            return np.array([0, 0])

    system = SystemSpec()
    gd = 1e-6
    system.forms = [ConstantGrowthForm("constant_growth_form", gd)]

    output_spec = OutputLastSpec()
    options = MCSolverOptions(output_spec)
    solver = MCSolver(system, options)

    init_n = create_normal_distribution_n(30e-6, 10e-6, count_density=1e6)
    initial_state = system.make_empty_state(concentration=1, temperature=0, n=[init_n])

    solve_time = 60

    input_ = Input()

    computed_states = solver.compute(initial_state, solve_time, input_)
    sdf = StateDataFrame(computed_states)
    n_last = sdf.n.iloc[0][0]
    expected = init_n.copy()
    expected[:, 0] += solve_time * gd
    assert np.allclose(expected, n_last)

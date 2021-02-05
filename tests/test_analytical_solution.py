import numpy as np

from crm.base.input import Input
from crm.base.output_spec import OutputLastSpec
from crm.base.state import State, create_normal_distribution_n
from crm.base.system_spec import SystemSpec, FormSpec
from crm.mcsolver import MCSolver, MCSolverOptions
from crm.utils.pandas import StateDataFrame


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


class ConstantGrowthNucleationForm(FormSpec):

    def __init__(self, name: str, gds=1e-6, nuc=1e2, ignore_volume=True):
        super().__init__(name)
        self.ignore_volume = ignore_volume
        self.nuc = nuc
        self.gds = gds

    def solubility(self, t: float) -> float:
        return 0.1

    def growth_rate(self, t: float, ss: float, n: np.ndarray = None, state: State = None) -> np.ndarray:
        return np.array(self.gds)

    def dissolution_rate(self, t: float, ss: float, n: np.ndarray = None, state: State = None) -> np.ndarray:
        return np.array(-self.gds)

    def nucleation_rate(self, t: float, ss: float, vf: float, state: State = None) -> np.ndarray:
        return np.array([self.nuc, 0])

    def volume_fraction(self, n: np.ndarray):
        if self.ignore_volume:
            return 0
        else:
            return super(ConstantGrowthNucleationForm, self).volume_fraction(n)


def test_pure_growth_and_dissolve():
    # define a model that has constant growth rate

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

    # dissolving
    computed_states[-1].concentration = 0.

    computed_states = solver.compute(computed_states[-1], solve_time, input_)
    sdf = StateDataFrame(computed_states)
    n_last = sdf.n.iloc[0][0]
    assert np.allclose(init_n, n_last)


def test_growth_and_nucleation():
    """
    Constant growth and nucleation. Expect a moving step function.
    :return:
    """
    system = SystemSpec()
    gd = 1e-6
    nuc = 1e2
    system.forms = [ConstantGrowthNucleationForm("constant_growth_form", gd, nuc, ignore_volume=True)]

    output_spec = OutputLastSpec()
    options = MCSolverOptions(output_spec)
    solver = MCSolver(system, options)

    initial_state = system.make_empty_state(concentration=1, temperature=0)

    solve_time = 60

    input_ = Input()

    computed_states = solver.compute(initial_state, solve_time, input_)
    sdf = StateDataFrame(computed_states)
    n_last = sdf.n.iloc[0][0]

    assert np.all(n_last[:, 1] - n_last[0, 1] == 0)
    assert np.isclose(n_last[:, 0].max(), gd * solve_time)

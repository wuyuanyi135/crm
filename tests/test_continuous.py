import io

import pytest
from pytest_html import extras

from crm.base.input import ContinuousInput
from crm.base.output_spec import OutputLastSpec
from crm.base.state import InletState
from crm.presets.hypothetical import Hypothetical1D
from crm.utils.csd import edges_to_center_grid
from crm.utils.pandas import StateDataFrame
from solvers.mcsolver import MCSolverOptions, MCSolver
from matplotlib import pyplot as plt
import base64


@pytest.mark.has_plot
def test_steady_state_plot(extra):
    system_spec = Hypothetical1D()
    inlet_state = system_spec.make_state(state_type=InletState, concentration=system_spec.forms[0].solubility(60),
                                         temperature=25, rt=300)
    input_ = ContinuousInput(inlet_state)

    initial_condition = system_spec.make_state(concentration=system_spec.forms[0].solubility(25), temperature=25)

    options = MCSolverOptions(output_spec=OutputLastSpec())
    solver = MCSolver(system_spec, options)

    output = solver.compute(init_state=initial_condition, solve_time=3600, input_=input_)
    sdf = StateDataFrame(output)

    n = sdf.get_csd()

    edge = n[1]
    grid = edges_to_center_grid(edge)
    data = n[0]["alpha", 0]

    plt.figure(figsize=(4, 4), dpi=300)
    plt.plot(grid * 1e6, data.iloc[0], ".")
    plt.yscale("log")
    plt.ylabel("Count")
    plt.xlabel("Size (micron)")
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    extra.append(extras.jpg(b64, ))

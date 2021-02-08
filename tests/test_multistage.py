import pytest

from crm.base.input import ConstantTemperatureInput, ContinuousInput
from crm.base.state import InletState
from crm.utils.multistage import Sequential, StageSpec
from crm.presets.hypothetical import Hypothetical1D, Hypothetical2D


def test_two_stage_1d():
    system_spec = Hypothetical1D()
    first_stage_input = ContinuousInput(
        system_spec.make_state(
            state_type=InletState,
            concentration=system_spec.forms[0].solubility(60),
            temperature=60,
            rt=600
        )
    )
    stages = [
        StageSpec(
            initial_condition=system_spec.make_state(
                concentration=system_spec.forms[0].solubility(40),
                temperature=40
            ),
            relative_volume=1,
            extra_input=ConstantTemperatureInput(40)
        ),
        StageSpec(
            initial_condition=system_spec.make_state(
                concentration=system_spec.forms[0].solubility(20),
                temperature=20
            ),
            relative_volume=1,
            extra_input=ConstantTemperatureInput(20)
        ),
    ]
    seq = Sequential(system_spec, first_stage_input, stages, 1)
    data = seq.compute(1800)
    assert len(data) == 2


def test_two_stage_2d():
    system_spec = Hypothetical2D()
    first_stage_input = ContinuousInput(
        system_spec.make_state(
            state_type=InletState,
            concentration=system_spec.forms[0].solubility(60),
            temperature=60,
            rt=600
        )
    )
    stages = [
        StageSpec(
            initial_condition=system_spec.make_state(
                concentration=system_spec.forms[0].solubility(40),
                temperature=40
            ),
            relative_volume=1,
            extra_input=ConstantTemperatureInput(40)
        ),
        StageSpec(
            initial_condition=system_spec.make_state(
                concentration=system_spec.forms[0].solubility(20),
                temperature=20
            ),
            relative_volume=1,
            extra_input=ConstantTemperatureInput(20)
        ),
    ]
    seq = Sequential(system_spec, first_stage_input, stages, 1)
    data = seq.compute(1800)
    assert len(data) == 2

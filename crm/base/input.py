from typing import Union, Tuple, List, Optional

from crm.base.state import State, InletState


class Input:
    """
    Mutate the state according to the input information.
    """

    def transform(self, state: State) -> State:
        # Changing the n is not allowed as input. It will break the conservation laws.
        return state

    def inlet(self, state: State) -> Optional[InletState]:
        """
        Inlet specification. The conditions affected by the time step will be specified here.
        :param state:
        :return: tuple of (state, rt) or None if no continuous IO
        """
        return None


class InputAssembler(Input):
    """
    Merge multiple input instances
    """

    def __init__(self, inputs: List[Input]):
        super().__init__()
        self.inputs = inputs

    def transform(self, state: State) -> State:
        for i in self.inputs:
            state = i.transform(state)
        return state

    def inlet(self, state: State) -> Union[InletState, None]:
        inlet_states = []
        for i in self.inputs:
            inlet_state = i.inlet(state)
            if inlet_state is not None:
                inlet_states.append(inlet_state)

        merged = inlet_states[0]
        for i in inlet_states[1:]:
            merged += i
        return merged


class ConstantTemperatureInput(Input):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def transform(self, state: State) -> State:
        state.temperature = self.temperature
        return state


class LinearTemperatureInput(Input):
    def __init__(self, start_temperature: float, end_temperature: float, abs_rate: float):
        assert abs_rate != 0

        self.abs_rate = abs(abs_rate)
        self.end_temperature = end_temperature
        self.start_temperature = start_temperature

        if start_temperature > end_temperature:
            self.rate = - self.abs_rate
            self.cooling = True
        else:
            self.rate = self.abs_rate
            self.cooling = False

    def transform(self, state: State) -> State:
        set_point = self.start_temperature + self.rate * state.time
        if (set_point <= self.end_temperature and self.cooling) or (
                set_point >= self.end_temperature and not self.cooling):
            set_point = self.end_temperature
        state.temperature = set_point
        return state


class ContinuousInput(Input):
    def __init__(self, inlet_state: InletState):
        """

        :param rt: residence time
        """
        super().__init__()
        self.inlet_state = inlet_state

    def inlet(self, state: State) -> Union[InletState, None]:
        return self.inlet_state

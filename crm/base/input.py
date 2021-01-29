from crm.base.state import State


class Input:
    """
    Mutate the state according to the input information.
    """

    def transform(self, state: State) -> State:
        pass


class ConstTemperatureInput(Input):
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
        set_point = self.start_temperature + self.abs_rate * state.time
        if (set_point <= self.end_temperature and self.cooling) or (
                set_point >= self.end_temperature and not self.cooling):
            set_point = self.end_temperature
        state.temperature = set_point
        return state

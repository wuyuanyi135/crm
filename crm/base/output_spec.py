from typing import List

from crm.base.state import State


class OutputSpec:
    """
    Base class for describing how the solver should make the output after computation
    """

    def should_update_output(self, state: State, end_time: float) -> bool:
        pass

    def update_output(self, state: State):
        """

        :param state:
        :return: if the output has been updated
        """
        pass

    def get_outputs(self) -> List[State]:
        pass


class OutputAllSpec(OutputSpec):
    def __init__(self):
        self.outputs = []

    def update_output(self, state: State):
        self.outputs.append(state)

    def should_update_output(self, state: State, end_time: float) -> bool:
        return True

    def get_outputs(self) -> List[State]:
        return self.outputs


class OutputLastSpec(OutputSpec):
    def __init__(self):
        self.output = None

    def should_update_output(self, state: State, end_time: float) -> bool:
        return end_time == state.time

    def update_output(self, state: State):
        self.output = state

    def get_outputs(self) -> List[State]:
        return [self.output]


class OutputIntervalSpec(OutputSpec):
    """
    Output every n seconds
    """

    def __init__(self, interval_second: float, include_last: bool):
        """

        :param interval_second:
        :param include_last: whether to include the last state if it is not on the interval.
        """
        self.include_last = include_last
        self.interval_second = interval_second
        self.outputs = []
        self.next_time = None

    def should_update_output(self, state: State, end_time: float) -> bool:
        return self.next_time is None or state.time >= self.next_time or (self.include_last and end_time == state.time)

    def update_output(self, state: State):
        self.outputs.append(state)
        self.next_time = state.time + self.interval_second

    def get_outputs(self) -> List[State]:
        return self.outputs


class OutputAtSpec(OutputSpec):
    """
    Output after given edge points.
    """

    def update_output(self, state: State):
        raise NotImplementedError()

    def should_update_output(self, state: State, end_time: float) -> bool:
        raise NotImplementedError()

    def get_outputs(self) -> List[State]:
        raise NotImplementedError()

"""Twin Calibration agent. Current version: no twin is fitted, so no deviation score is given.

Writes: state.deviation, state.twin_status.
"""
from typing import override

from .state import Agent, PatientState


class TwinAgent(Agent):
    name = "twin"

    @override
    def step(self, state: PatientState) -> None:
        state.deviation = None  # no twin yet, so Detection works without the deviation score
        state.twin_status = "uncalibrated"

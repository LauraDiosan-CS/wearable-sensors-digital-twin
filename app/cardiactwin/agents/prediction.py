"""Physiological Prediction agent. Current version: makes no forecast.

Writes: state.forecast, and heart-rate alerts to state.alerts.
"""
from typing import override

from .state import Agent, PatientState


class PredictionAgent(Agent):
    name = "prediction"

    @override
    def step(self, state: PatientState) -> None:
        pass

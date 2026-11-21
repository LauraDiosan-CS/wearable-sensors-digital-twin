"""Arrhythmia Detection agent. Current version: labels every beat N.

Writes: state.labels, state.probs, and events to state.alerts.
"""
from typing import override

import numpy as np

from .state import Agent, Label, PatientState

CLASSES: tuple[Label, ...] = ("N", "S", "V", "F")  # column order of state.probs


class DetectionAgent(Agent):
    name = "detection"

    @override
    def step(self, state: PatientState) -> None:
        n = len(state.beats)
        state.probs = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
        state.labels = ["N"] * n
        if n:
            state.decide(self.name, "label", f"all {n} beats labelled N")

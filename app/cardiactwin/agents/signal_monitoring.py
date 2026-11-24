"""Signal Monitoring agent. Current version: accepts every window and only band-passes it.

Writes: state.quality, state.usable, state.beats.
"""
from typing import override

import numpy as np

from .. import data
from .state import BEAT_LEN, Agent, PatientState


class SignalMonitoringAgent(Agent):
    name = "signal"

    @override
    def step(self, state: PatientState) -> None:
        assert state.window is not None
        w = state.window
        clean = data.bandpass(w.signal)
        state.beats = (np.stack([clean[r - data.PRE:r + data.POST] for r in w.r_peaks]) if len(w.r_peaks)
                       else np.empty((0, BEAT_LEN)))
        state.quality = 1.0
        state.usable = True
        state.decide(self.name, "accept", "no quality check, band-pass only")

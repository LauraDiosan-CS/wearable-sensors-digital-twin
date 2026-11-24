"""Twin Calibration agent.

Cold start: collect the beats of the first CALIB_S seconds, then fit the McSharry twin to their median beat.
From then on every beat gets a deviation score: its distance from the twin, relative to the patient's usual
beat-to-beat variation, so a typical beat scores about 1.

Writes: state.deviation, state.twin_status.
"""
from typing import override

import numpy as np

from .. import twin
from .state import Agent, FloatArray, PatientState

CALIB_S = 300.0  # seconds of signal used for the first fit
MIN_BEATS = 100  # fewest beats needed for a fit


class TwinAgent(Agent):
    name = "twin"

    def __init__(self) -> None:
        self.calib_beats: list[FloatArray] = []     # cold-start buffer: beats of each window
        self.calib_rr: list[FloatArray] = []        # and their RR intervals
        self.twin_beat: FloatArray | None = None    # the twin's beat in mV (252 samples), None until fitted
        self.params: FloatArray | None = None       # the 15 McSharry parameters behind it
        self.rr = 0.0                               # RR interval (s) the twin is drawn at
        self.sigma = 1.0                            # median RMS distance (mV) of this patient's beats from the twin

    @override
    def step(self, state: PatientState) -> None:
        if self.twin_beat is None:
            self._cold_start(state)
        self._score(state)

    def _cold_start(self, state: PatientState) -> None:
        """Collect beats until CALIB_S seconds have passed, then fit the twin once."""
        assert state.window is not None
        self.calib_beats.append(state.beats)
        self.calib_rr.append(state.window.rr[:, 0]) # previous RR intervals of the beats in this window
        if state.t < CALIB_S:
            return
        beats = np.concatenate(self.calib_beats)
        if len(beats) < MIN_BEATS:
            state.decide(self.name, "wait", f"only {len(beats)} usable beats after {state.t:.0f} s, need {MIN_BEATS}")
            return
        self._fit(beats, np.concatenate(self.calib_rr))
        self.calib_beats, self.calib_rr = [], []
        state.decide(self.name, "calibrate", f"fitted on the median of {len(beats)} beats from the first {state.t:.0f} s")

    def _fit(self, beats: FloatArray, rr: FloatArray) -> None:
        """Fit the twin to the median of `beats` and measure how far these beats usually are from it."""
        template = np.median(beats, axis=0)
        self.rr = float(np.median(rr))
        self.params, self.twin_beat = twin.fit(template, self.rr)
        self.sigma = float(np.median(twin.rms(beats - self.twin_beat)))

    def _score(self, state: PatientState) -> None:
        """Deviation of every beat in this window from the twin, in units of the patient's usual variation."""
        if self.twin_beat is None:
            state.deviation, state.twin_status = None, "uncalibrated"
            return
        state.twin_status = "personal"
        state.deviation = twin.rms(state.beats - self.twin_beat) / self.sigma

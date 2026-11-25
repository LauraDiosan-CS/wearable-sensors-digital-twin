"""Twin Calibration agent.

Cold start: collect the beats of the first CALIB_S seconds, then fit the McSharry twin to their median beat.
From then on every beat gets a deviation score: its distance from the twin, relative to the patient's usual
beat-to-beat variation, so a typical beat scores about 1.

Learning: beats that Arrhythmia Detection labeled N with high confidence go into a buffer of recent normal beats, and a
CUSUM adds up how far their scores stay above 1.
Depending on the policy, the twin is refitted:
- never
- on a fixed schedule (on the whole buffer)
- or when the CUSUM signals that the patient's normal beat drifted (on the most recent beats only, since older ones no
  longer describe the patient).

Writes: state.deviation, state.twin_status, and one review alert per drift episode.
"""
from typing import Literal, TypedDict, override

import numpy as np

from .. import twin
from .state import Agent, FloatArray, PatientState

type Policy = Literal["none", "scheduled", "drift"]

CALIB_S = 300.0         # seconds of signal used for the first fit
MIN_BEATS = 100         # fewest beats needed for the first fit
MIN_REFIT_BEATS = 50    # fewest beats needed for a refit
P_NORMAL = 0.9          # a beat must be labeled N with at least this probability to count as normal
BUFFER_S = 300.0        # how many seconds of recent normal beats are kept for refitting
RECAL_EVERY_S = 600.0   # "scheduled" policy: refit this often
RECENT_S = 60.0         # "drift" policy: refit on the normal beats of the last RECENT_S seconds only
ALERT_COOLDOWN_S = 600.0  # "drift" policy: at most one alert in this time; later refits are only logged


class FitRecord(TypedDict):
    t: float            # time of the fit (s)
    reason: str
    n_beats: int
    params: FloatArray  # the 15 McSharry parameters
    sigma: float


class TwinAgent(Agent):
    """
    Twin Calibration agent.

    Args:
        policy: "none", "scheduled", or "drift"
        cusum_k: Only for "drift" policy: score above 1 + K adds to the CUSUM
        cusum_h: Only for "drift" policy: refit when the CUSUM passes this threshold.
                 Defaults are tuned on DS1, see scripts/tune_cusum.py.
    """
    name = "twin"

    def __init__(self, policy: Policy = "drift", cusum_k: float = 0.5, cusum_h: float = 40.0) -> None:
        self.policy = policy
        self.cusum_k, self.cusum_h = cusum_k, cusum_h  # constructor arguments so experiments can vary them
        self.calib_beats: list[FloatArray] = []     # cold-start buffer: beats of each window
        self.calib_rr: list[FloatArray] = []        # and their RR intervals
        self.twin_beat: FloatArray | None = None    # the twin's beat in mV (252 samples), None until fitted
        self.params: FloatArray | None = None       # the 15 McSharry parameters behind it
        self.rr = 0.0                               # RR interval (s) the twin is drawn at
        self.sigma = 1.0                            # median RMS distance (mV) of this patient's beats from the twin
        self.normal: list[tuple[float, FloatArray, FloatArray]] = []  # (time, beats, RR) of confident N beats
        self.cusum = 0.0
        self.fitted_at = 0.0                        # time of the last fit (s)
        self.alerted_at = -np.inf                   # time of the last drift alert (s)
        self.history: list[FitRecord] = []          # every fit, for analysis and the report

    @override
    def step(self, state: PatientState) -> None:
        if self.twin_beat is None:
            self._cold_start(state)
        else:
            self._learn(state)
            self._recalibrate(state)
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
        reason = f"fitted on the median of {len(beats)} beats from the first {state.t:.0f} s"
        self._fit(beats, np.concatenate(self.calib_rr), state.t, reason)
        self.calib_beats, self.calib_rr = [], []
        state.decide(self.name, "calibrate", reason)

    def _learn(self, state: PatientState) -> None:
        """Keep the previous window's confident N beats and update the CUSUM with their scores."""
        prev = state.prev
        if prev is None or not prev["usable"] or prev["deviation"] is None or not len(prev["labels"]):
            return
        normal = (np.array(prev["labels"]) == "N") & (prev["probs"][:, 0] >= P_NORMAL)
        if normal.any():
            self.normal.append((prev["window"].t_end, prev["beats"][normal], prev["window"].rr[normal, 0]))
        for d in prev["deviation"][normal]:
            self.cusum = max(0.0, self.cusum + d - 1 - self.cusum_k)
        self.normal = [entry for entry in self.normal if entry[0] > state.t - BUFFER_S]

    def _recalibrate(self, state: PatientState) -> None:
        """Refit the twin on recent normal beats if the policy says so."""
        match self.policy:
            case "none":
                return
            case "scheduled":
                if state.t - self.fitted_at < RECAL_EVERY_S:
                    return
                reason = f"scheduled, {state.t - self.fitted_at:.0f} s since the last fit"
                since = state.t - BUFFER_S
            case "drift":
                if self.cusum <= self.cusum_h:
                    return
                reason = f"drift, CUSUM {self.cusum:.1f} > {self.cusum_h:g}"
                since = state.t - RECENT_S
        recent = [(beats, rr) for t, beats, rr in self.normal if t > since]
        n = sum(len(beats) for beats, _ in recent)
        if n < MIN_REFIT_BEATS:
            state.decide(self.name, "skip", f"{reason}, but only {n} confident normal beats, need {MIN_REFIT_BEATS}")
            return
        beats = np.concatenate([beats for beats, _ in recent])
        rr = np.concatenate([rr for _, rr in recent])
        reason += f"; refitted on the median of {n} confident normal beats from the last {state.t - since:.0f} s"
        self._fit(beats, rr, state.t, reason)
        self.cusum = 0.0
        state.decide(self.name, "recalibrate", reason)
        if self.policy == "drift" and state.t - self.alerted_at >= ALERT_COOLDOWN_S:
            self.alerted_at = state.t
            state.alert(self.name, "review", "the patient's normal beat has changed; the twin was recalibrated")

    def _fit(self, beats: FloatArray, rr: FloatArray, t: float, reason: str) -> None:
        """Fit the twin to the median of `beats` and measure how far these beats usually are from it."""
        template = np.median(beats, axis=0)
        self.rr = float(np.median(rr))
        self.params, self.twin_beat = twin.fit(template, self.rr)
        self.sigma = float(np.median(twin.rms(beats - self.twin_beat)))
        self.fitted_at = t
        self.history.append(FitRecord(t=t, reason=reason, n_beats=len(beats), params=self.params, sigma=self.sigma))

    def _score(self, state: PatientState) -> None:
        """Deviation of every beat in this window from the twin, in units of the patient's usual variation."""
        if self.twin_beat is None:
            state.deviation, state.twin_status = None, "uncalibrated"
            return
        state.twin_status = "personal"
        state.deviation = twin.rms(state.beats - self.twin_beat) / self.sigma

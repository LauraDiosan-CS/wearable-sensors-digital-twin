"""Tests of the Twin Calibration agent on synthetic ECG beats (no downloaded data needed).

usage (from the app/ folder):  pytest tests
"""
import sys
from collections.abc import Callable
from pathlib import Path
from typing import override

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import data, twin  # noqa: E402
from cardiactwin.agents import Agent, PatientState, Window, step_window  # noqa: E402
from cardiactwin.agents.detection import DetectionAgent  # noqa: E402
from cardiactwin.agents.signal_monitoring import SignalMonitoringAgent  # noqa: E402
from cardiactwin.agents.state import FloatArray, Label  # noqa: E402
from cardiactwin.agents.twin_agent import CALIB_S, RECAL_EVERY_S, Policy, TwinAgent  # noqa: E402

RR = 0.8  # seconds between beats (75 bpm)


def make_beat(params: FloatArray) -> FloatArray:
    """A beat from the McSharry model, scaled to about 1.2 mV peak to peak like a real one."""
    z = twin.model_beat(params, RR)
    return z / np.ptp(z) * 1.2


NORMAL = make_beat(twin.default_params(RR))


def abnormal_beat() -> FloatArray:
    """A ventricular-looking beat: three times wider QRS, inverted T wave."""
    p = twin.default_params(RR).copy()
    p[11:14] *= 3   # widths of Q, R, S
    p[9] *= -1      # strength of T
    return make_beat(p)


def make_window(t_start: float, beat: FloatArray, rng: np.random.Generator, noise: float = 0.02) -> Window:
    """10 s of noisy signal with `beat` every RR seconds, cut like the orchestrator does."""
    fs = data.FS
    n = int(10 / RR)
    signal = rng.normal(0, noise, 10 * fs + data.PRE + data.POST)
    r_peaks = data.PRE + (np.arange(n) * RR * fs + 0.3 * fs).astype(np.int64)
    for r in r_peaks:
        signal[r - data.PRE:r + data.POST] += beat
    return Window(record=0, t_start=t_start, t_end=t_start + 10, start=int(t_start * fs) - data.PRE,
                  signal=signal, r_peaks=r_peaks, rr=np.full((n, 3), RR))


def calibrated() -> tuple[PatientState, list, np.random.Generator]:
    """Run Signal Monitoring + Twin through the whole cold start on normal beats."""
    rng = np.random.default_rng(0)
    state, agents = PatientState(0), [SignalMonitoringAgent(), TwinAgent()]
    for k in range(int(CALIB_S / 10)):
        step_window(state, make_window(10 * k, NORMAL, rng), agents)
    return state, agents, rng


def test_no_score_during_cold_start() -> None:
    rng = np.random.default_rng(0)
    state, agents = PatientState(0), [SignalMonitoringAgent(), TwinAgent()]
    for k in range(int(CALIB_S / 10) - 1):
        step_window(state, make_window(10 * k, NORMAL, rng), agents)
        assert state.deviation is None and state.twin_status == "uncalibrated"


def test_calibrates_after_cold_start_and_normal_beats_score_about_one() -> None:
    state, _, _ = calibrated()
    assert state.twin_status == "personal"
    assert [row["action"] for row in state.log if row["agent"] == "twin"] == ["calibrate"]
    assert state.deviation is not None and 0.5 < np.median(state.deviation) < 2


def test_abnormal_beat_scores_far_above_normal() -> None:
    state, agents, rng = calibrated()
    step_window(state, make_window(CALIB_S, abnormal_beat(), rng), agents)
    assert state.deviation is not None and np.median(state.deviation) > 5


# --- learning and recalibration -----------------------------------------------------------------------


def flattened_t_beat(factor: float) -> FloatArray:
    """A normal beat whose T wave is scaled by `factor` (1 = unchanged), to simulate slow drift."""
    p = twin.default_params(RR).copy()
    p[9] *= factor  # strength of T
    return make_beat(p)


def run(policy: Policy, beat_at: Callable[[float], FloatArray], minutes: int) -> tuple[PatientState, TwinAgent]:
    """Signal Monitoring + Twin + the placeholder Detection (every beat N, probability 1) for `minutes`."""
    rng = np.random.default_rng(0)
    twin_agent = TwinAgent(policy)
    state, agents = PatientState(0), [SignalMonitoringAgent(), twin_agent, DetectionAgent()]
    for k in range(minutes * 6):
        step_window(state, make_window(10 * k, beat_at(10 * k), rng), agents)
    return state, twin_agent


def drifting(t: float) -> FloatArray:
    """Normal beats; from minute 10 to 20 the T wave shrinks linearly to 30% and stays there."""
    return flattened_t_beat(1 - 0.7 * float(np.clip((t - 600) / 600, 0, 1)))


def recalibrations(state: PatientState) -> list[str]:
    return [row["reason"] for row in state.log if row["agent"] == "twin" and row["action"] == "recalibrate"]


class MixedLabels(Agent):
    """Labels beats in turn: N (sure), N (unsure), V (sure)."""
    name = "detection"

    @override
    def step(self, state: PatientState) -> None:
        n = len(state.beats)
        choices: list[tuple[Label, list[float]]] = [("N", [0.95, 0, 0.05, 0]), ("N", [0.5, 0.2, 0.3, 0]),
                                                    ("V", [0.05, 0, 0.95, 0])]
        state.labels = [choices[i % 3][0] for i in range(n)]
        state.probs = np.array([choices[i % 3][1] for i in range(n)])


def test_learns_only_from_confident_normal_beats() -> None:
    state, (signal, twin_agent), rng = calibrated()
    agents = [signal, twin_agent, MixedLabels()]
    step_window(state, make_window(CALIB_S, NORMAL, rng), agents)
    step_window(state, make_window(CALIB_S + 10, NORMAL, rng), agents)  # learns from the window before
    assert isinstance(twin_agent, TwinAgent)
    n_kept = sum(len(beats) for _, beats, _ in twin_agent.normal)
    assert n_kept == len([i for i in range(len(state.prev["labels"])) if i % 3 == 0])  # type: ignore[index]


def test_drift_policy_ignores_a_stable_patient() -> None:
    state, _ = run("drift", lambda t: NORMAL, minutes=20)
    assert recalibrations(state) == [] and state.alerts == []


def test_drift_policy_follows_a_drifting_patient() -> None:
    state, _ = run("drift", drifting, minutes=25)
    reasons = recalibrations(state)
    assert reasons and all(r.startswith("drift") and r.endswith("from the last 60 s") for r in reasons)
    assert len(state.alerts) == 1 and state.alerts[0]["level"] == "review"  # one alert for the whole episode
    assert state.deviation is not None and np.median(state.deviation) < 2  # back to "normal for this patient"


def test_without_recalibration_drift_keeps_scores_high() -> None:
    state, _ = run("none", drifting, minutes=25)
    assert recalibrations(state) == []
    assert state.deviation is not None and np.median(state.deviation) > 2


def test_scheduled_policy_refits_at_fixed_times() -> None:
    state, twin_agent = run("scheduled", lambda t: NORMAL, minutes=25)
    times = [fit["t"] for fit in twin_agent.history]
    assert times == [CALIB_S, CALIB_S + RECAL_EVERY_S, CALIB_S + 2 * RECAL_EVERY_S]
    assert len(recalibrations(state)) == 2

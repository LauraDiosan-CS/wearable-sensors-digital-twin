"""Tests of the Twin Calibration agent on synthetic ECG beats (no downloaded data needed).

usage (from the app/ folder):  pytest tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import data, twin  # noqa: E402
from cardiactwin.agents import PatientState, Window, step_window  # noqa: E402
from cardiactwin.agents.signal_monitoring import SignalMonitoringAgent  # noqa: E402
from cardiactwin.agents.state import FloatArray  # noqa: E402
from cardiactwin.agents.twin_agent import CALIB_S, TwinAgent  # noqa: E402

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

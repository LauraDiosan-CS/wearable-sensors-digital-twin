"""Tests of the agent contract and orchestrator on a hand-made window (no downloaded data needed).

usage (from the app/ folder):  pytest tests
"""
import sys
from pathlib import Path
from typing import override

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cardiactwin import data  # noqa: E402
from cardiactwin.agents import Agent, PatientState, Window, default_agents, step_window  # noqa: E402


def make_window(t_start: float = 0.0, n_beats: int = 12, rr: float = 0.8) -> Window:
    """A 10 s window of spikes every `rr` seconds, with the same margins the orchestrator uses."""
    fs = data.FS
    signal = np.zeros(10 * fs + data.PRE + data.POST)
    r_peaks = data.PRE + (np.arange(n_beats) * rr * fs + 0.3 * fs).astype(np.int64)
    for r in r_peaks:
        signal[r - 4:r + 5] = 1.0
    rrs = np.full((n_beats, 3), rr)
    return Window(record=0, t_start=t_start, t_end=t_start + 10, start=int(t_start * fs) - data.PRE,
                  signal=signal, r_peaks=r_peaks, rr=rrs)


def test_one_window_through_all_agents() -> None:
    state, w = PatientState(0), make_window(t_start=10)
    step_window(state, w, default_agents())
    assert state.beats.shape == (12, data.PRE + data.POST)
    assert len(state.labels) == 12 and state.probs.shape == (12, 4)
    assert len(state.beat_log) == 12
    assert state.log and all({"t", "agent", "action", "reason"} <= set(row) for row in state.log)


def test_previous_window_is_kept() -> None:
    state, agents = PatientState(0), default_agents()
    step_window(state, make_window(t_start=10), agents)
    step_window(state, make_window(t_start=20, n_beats=11), agents)
    assert state.prev is not None
    assert state.prev["window"].t_start == 10 and len(state.prev["labels"]) == 12
    assert len(state.labels) == 11 and len(state.beat_log) == 23


def test_agent_without_step_cannot_be_created() -> None:
    class Incomplete(Agent):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


class RejectAll(Agent):
    name = "signal"

    @override
    def step(self, state: PatientState) -> None:
        state.usable = False
        state.decide(self.name, "reject", "test")


class Spy(Agent):
    """Counts how often it is called."""
    name = "spy"

    def __init__(self) -> None:
        self.calls = 0

    @override
    def step(self, state: PatientState) -> None:
        self.calls += 1


class TwinSpy(Spy):
    name = "twin"


class DetectionSpy(Spy):
    name = "detection"


class PredictionSpy(Spy):
    name = "prediction"


class ClinicalSpy(Spy):
    name = "clinical"


def test_rejected_window_skips_middle_agents() -> None:
    middle: list[Spy] = [TwinSpy(), DetectionSpy(), PredictionSpy()]
    clinical = ClinicalSpy()
    state = PatientState(0)
    step_window(state, make_window(t_start=10), [RejectAll(), *middle, clinical])
    assert all(a.calls == 0 for a in middle) and clinical.calls == 1
    assert {row["label"] for row in state.beat_log} == {"-"}

"""Orchestrator: replays a record in time order and runs the agents in a fixed order on every window."""
from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from .. import data
from .clinical import ClinicalAgent
from .detection import DetectionAgent
from .prediction import PredictionAgent
from .signal_monitoring import SignalMonitoringAgent
from .state import Agent, PatientState, Window
from .twin_agent import TwinAgent

type Truth = NDArray[np.str_]  # true AAMI class of each beat, for evaluation only

WINDOW_S = 10
ALWAYS_RUN = frozenset({"signal", "clinical"})  # these also run on rejected windows


def default_agents() -> list[Agent]:
    """A fresh set of agents for one patient, in the order they run."""
    return [SignalMonitoringAgent(), TwinAgent(), DetectionAgent(), PredictionAgent(), ClinicalAgent()]


def replay(record: int, window_s: int = WINDOW_S) -> Iterator[tuple[Window, Truth]]:
    """Yield (Window, true labels) for consecutive windows of a record. The labels are for evaluation only."""
    d = data.extract_beats(record)
    sig = data.load_signal(record)
    r = np.round(d["time_s"] * data.FS).astype(np.int64)
    rr = np.stack([d["pre_rr"], d["post_rr"], d["local_rr"]], axis=1).astype(np.float64)
    truth = np.array(data.CLASSES)[d["labels"]]
    n_windows = int(np.ceil(len(sig) / data.FS / window_s))
    for k in range(n_windows):
        t0, t1 = k * window_s, (k + 1) * window_s
        m = (d["time_s"] >= t0) & (d["time_s"] < t1)
        s0 = max(0, int(t0 * data.FS) - data.PRE)
        s1 = min(len(sig), int(t1 * data.FS) + data.POST)
        yield Window(record, t0, t1, s0, sig[s0:s1], r[m] - s0, rr[m]), truth[m]


def step_window(state: PatientState, window: Window, agents: Sequence[Agent]) -> None:
    """Run all agents on one window, in order. A rejected window skips the middle agents."""
    state.new_window(window)
    for agent in agents:
        if state.usable or agent.name in ALWAYS_RUN:
            agent.step(state)
    state.record_beats()


def run_record(record: int, agents: Sequence[Agent] | None = None,
               window_s: int = WINDOW_S) -> tuple[PatientState, Sequence[Agent], Truth]:
    """Run the whole system on one record. Returns (state, agents, true label of every beat in beat_log)."""
    agents = agents if agents is not None else default_agents()
    state = PatientState(record)
    truth: list[str] = []
    for window, labels in replay(record, window_s):
        step_window(state, window, agents)
        truth.extend(labels)
    return state, agents, np.array(truth)

"""The shared contract between the agents.

Every agent has one method, `step(state)`: it reads what it needs from the PatientState, writes only its
own fields and logs each decision with `state.decide(...)`. Private memory (e.g. the twin's parameters)
stays inside the agent; anything another agent needs goes into the state. Agents never see true labels.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

from .. import data

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type Label = Literal["N", "S", "V", "F", "?", "-"]  # "?" = abstained, "-" = window rejected
type AlertLevel = Literal["info", "review", "urgent"]
type TwinStatus = Literal["uncalibrated", "personal"]

BEAT_LEN = data.PRE + data.POST  # 252 samples per beat
N_CLASSES = 4  # probabilities of N, S, V, F


class Forecast(TypedDict):
    hr_now: float  # bpm
    hr_next_min: float  # bpm
    alert: bool


class Alert(TypedDict):
    t: float
    agent: str
    level: AlertLevel
    text: str


class Decision(TypedDict):
    t: float
    agent: str
    action: str
    reason: str


class BeatRow(TypedDict):
    t: float
    label: Label
    deviation: float | None
    quality: float | None


@dataclass(frozen=True)
class Window:
    """10 s of one record, handed out by the orchestrator in time order."""
    record: int
    t_start: float  # seconds from the start of the record
    t_end: float
    start: int  # index of signal[0] in the whole record (samples)
    signal: FloatArray  # raw MLII in mV, with PRE- / POST-margins so every beat can be cut
    r_peaks: IntArray  # beat positions in `signal` (samples), from the annotations
    rr: FloatArray  # (n_beats, 3): previous RR, next RR, local mean RR, in seconds

    @property
    def beat_times(self) -> FloatArray:
        """Time of every beat's R peak, in seconds from the start of the record."""
        return (self.start + self.r_peaks) / data.FS


class WindowResult(TypedDict):
    """What the agents produced for one window; the previous one is kept in `PatientState.prev`."""
    window: Window
    twin_status: TwinStatus
    quality: float | None
    usable: bool
    beats: FloatArray
    deviation: FloatArray | None
    labels: list[Label]
    probs: FloatArray
    forecast: Forecast | None


def _no_beats() -> FloatArray:
    return np.empty((0, BEAT_LEN))


def _no_probs() -> FloatArray:
    return np.empty((0, N_CLASSES))


@dataclass
class PatientState:
    """Everything the agents share about one patient."""
    record: int
    window: Window | None = None
    prev: WindowResult | None = None  # the previous window's results (read-only)
    # written by Signal Monitoring
    quality: float | None = None  # 0..1
    usable: bool = True  # False = window rejected, the middle agents skip it
    beats: FloatArray = field(default_factory=_no_beats)  # (n_beats, 252) cleaned beats
    # written by Twin Calibration
    deviation: FloatArray | None = None  # (n_beats,) deviation score, None until calibrated
    twin_status: TwinStatus = "uncalibrated"
    # written by Arrhythmia Detection
    labels: list[Label] = field(default_factory=list)  # one per beat
    probs: FloatArray = field(default_factory=_no_probs)  # (n_beats, 4): probabilities of N, S, V, F
    # written by Physiological Prediction
    forecast: Forecast | None = None
    # memory over the whole record (appended, never cleared)
    alerts: list[Alert] = field(default_factory=list)
    log: list[Decision] = field(default_factory=list)
    beat_log: list[BeatRow] = field(default_factory=list)  # one row per beat, for evaluation

    @property
    def t(self) -> float:
        """Current time: the end of the current window (seconds)."""
        return self.window.t_end if self.window is not None else 0.0

    def decide(self, agent: str, action: str, reason: str) -> None:
        """Record one decision in the audit log."""
        self.log.append(Decision(t=self.t, agent=agent, action=action, reason=reason))

    def alert(self, agent: str, level: AlertLevel, text: str) -> None:
        """Raise an alert for the clinician."""
        self.alerts.append(Alert(t=self.t, agent=agent, level=level, text=text))

    def new_window(self, window: Window) -> None:
        """Move the current results to `prev`, clear the per-window fields and take the next window."""
        if self.window is not None:
            self.prev = WindowResult(
                window=self.window, twin_status=self.twin_status, quality=self.quality, usable=self.usable,
                beats=self.beats, deviation=self.deviation, labels=self.labels, probs=self.probs,
                forecast=self.forecast)
        self.window = window
        self.quality, self.usable, self.beats, self.deviation = None, True, _no_beats(), None
        self.labels, self.probs, self.forecast = [], _no_probs(), None

    def record_beats(self) -> None:
        """Append this window's beats to beat_log (called by the orchestrator after all agents)."""
        if self.window is None:
            return
        for i, t in enumerate(self.window.beat_times):
            label: Label = self.labels[i] if self.usable and i < len(self.labels) else "-"
            self.beat_log.append(BeatRow(
                t=float(t), label=label, quality=self.quality,
                deviation=float(self.deviation[i]) if self.deviation is not None else None))


class Agent(ABC):
    """One instance per patient; subclasses set `name` and implement `step`."""
    name: ClassVar[str]

    @abstractmethod
    def step(self, state: PatientState) -> None:
        """Read from the state, write this agent's own fields, log every decision."""

"""Clinical Recommendation agent.

Every window (also rejected ones) it notes how much signal was usable and the heart rate. At the end of a
record, `report()` collects the facts from the state into a structured report and `explain()` turns the report
into a short text for the clinician. The report only restates what the other agents found; it does not diagnose.

Current version: the priority is always routine (priority rules come next).
"""
from collections import Counter
from typing import Literal, TypedDict, override

import numpy as np

from .state import Agent, AlertLevel, Label, PatientState, TwinStatus

type Priority = Literal["routine", "review", "urgent"]


class HeartRate(TypedDict):
    median: float | None  # bpm, over the usable 10 s windows
    low: float | None
    high: float | None


class Report(TypedDict):
    record: int
    minutes: float
    beats: dict[Label, int]          # number of beats per label ("?" = not sure, "-" = signal rejected)
    v_burden_pct: float              # share of the labelled beats (N, S, V, F) that are V
    unsure_pct: float                # share of the usable beats labelled "?"
    signal_rejected_pct: float       # share of the windows Signal Monitoring rejected
    heart_rate: HeartRate
    alerts: dict[AlertLevel, int]    # alerts raised by the other agents, per level
    twin_status: TwinStatus
    twin_refits: int
    priority: Priority
    reasons: list[str]               # why this priority


class ClinicalAgent(Agent):
    name = "clinical"

    def __init__(self) -> None:
        self.windows = 0
        self.rejected = 0
        self.heart_rates: list[float] = []  # mean heart rate of each usable window (bpm)

    @override
    def step(self, state: PatientState) -> None:
        assert state.window is not None
        self.windows += 1
        if not state.usable:
            self.rejected += 1
        elif len(state.window.rr):
            self.heart_rates.append(float(60.0 / np.mean(state.window.rr[:, 0])))

    def report(self, state: PatientState) -> Report:
        """Structured report: only facts taken from the state."""
        beats = Counter(row["label"] for row in state.beat_log)
        labelled = sum(beats[c] for c in ("N", "S", "V", "F"))
        usable = labelled + beats["?"]
        hr = np.array(self.heart_rates)
        return Report(
            record=state.record,
            minutes=round(state.t / 60, 1),
            beats=dict(beats),
            v_burden_pct=round(100 * beats["V"] / labelled, 1) if labelled else 0.0,
            unsure_pct=round(100 * beats["?"] / usable, 1) if usable else 0.0,
            signal_rejected_pct=round(100 * self.rejected / max(self.windows, 1), 1),
            heart_rate=HeartRate(median=round(float(np.median(hr))) if len(hr) else None,
                                 low=round(float(hr.min())) if len(hr) else None,
                                 high=round(float(hr.max())) if len(hr) else None),
            alerts=dict(Counter(a["level"] for a in state.alerts)),
            twin_status=state.twin_status,
            twin_refits=sum(1 for row in state.log if row["agent"] == "twin" and row["action"] == "recalibrate"),
            priority="routine",
            reasons=[],
        )

    def explain(self, report: Report) -> str:
        """Readable text for the clinician, from a fixed template."""
        hr = report["heart_rate"]
        lines = [
            f"Record {report['record']}, {report['minutes']} min. Priority: {report['priority']}.",
            f"Beats: {', '.join(f'{k}: {v}' for k, v in sorted(report['beats'].items()))}; "
            f"ventricular beats {report['v_burden_pct']}% of the labelled beats.",
            f"Heart rate {hr['median']} bpm (range {hr['low']}-{hr['high']})." if hr["median"] is not None
            else "Heart rate unknown.",
            f"Signal rejected: {report['signal_rejected_pct']}% of the time; "
            f"beats left unlabelled (not sure): {report['unsure_pct']}%.",
            f"Twin: {report['twin_status']}, refitted {report['twin_refits']} times.",
        ]
        lines += [f"Reason: {r}." for r in report["reasons"]]
        return "\n".join(lines)

"""Clinical Recommendation agent.

Every window (also rejected ones) it notes how much signal was usable and the heart rate. At the end of a
record, `report()` collects the facts from the state into a structured report and `explain()` turns the report
into a short text for the clinician. The report only restates what the other agents found; it does not diagnose.

Priority, with the reasons written into the report and the decision log:
- urgent: another agent raised an urgent alert (e.g. a run of ventricular beats, a heart-rate alert);
- review: a review alert (e.g. the patient's normal beat changed), a high ventricular burden, many beats
  Detection was not sure about, or too much rejected signal to trust the rest;
- routine: none of the above.
"""
from collections import Counter
from typing import Literal, TypedDict, override

import numpy as np

from .state import Agent, AlertLevel, Label, PatientState, TwinStatus

type Priority = Literal["routine", "review", "urgent"]

V_BURDEN_REVIEW_PCT = 10.0   # ventricular beats in at least this share of labelled beats -> review
UNSURE_REVIEW_PCT = 10.0     # at least this share of usable beats labelled "?" -> review
REJECTED_REVIEW_PCT = 20.0   # at least this share of windows rejected -> review (too little usable signal)
MAX_ALERTS_LISTED = 3        # how many alerts of one level are quoted in the reasons


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
        """Structured report: facts taken from the state, and the priority decided from them."""
        beats = Counter(row["label"] for row in state.beat_log)
        labelled = sum(beats[c] for c in ("N", "S", "V", "F"))
        usable = labelled + beats["?"]
        hr = np.array(self.heart_rates)
        report = Report(
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
        report["priority"], report["reasons"] = self._priority(report, state)
        state.decide(self.name, report["priority"], "; ".join(report["reasons"]))
        return report

    def _priority(self, report: Report, state: PatientState) -> tuple[Priority, list[str]]:
        """The highest level that applies, with every reason for it."""
        urgent = [f"{a['agent']}: {a['text']}" for a in state.alerts if a["level"] == "urgent"]
        if urgent:
            return "urgent", self._quote(urgent)
        review = self._quote([f"{a['agent']}: {a['text']}" for a in state.alerts if a["level"] == "review"])
        if report["v_burden_pct"] >= V_BURDEN_REVIEW_PCT:
            review.append(f"ventricular beats are {report['v_burden_pct']}% of the labelled beats "
                          f"(review from {V_BURDEN_REVIEW_PCT:g}%)")
        if report["unsure_pct"] >= UNSURE_REVIEW_PCT:
            review.append(f"Detection was not sure about {report['unsure_pct']}% of the beats "
                          f"(review from {UNSURE_REVIEW_PCT:g}%)")
        if report["signal_rejected_pct"] >= REJECTED_REVIEW_PCT:
            review.append(f"{report['signal_rejected_pct']}% of the signal was rejected "
                          f"(review from {REJECTED_REVIEW_PCT:g}%)")
        if review:
            return "review", review
        return "routine", ["no alerts, and all measures below the review thresholds"]

    @staticmethod
    def _quote(alerts: list[str]) -> list[str]:
        """The first MAX_ALERTS_LISTED different alerts (repeats counted), plus how many more there were."""
        counts = Counter(alerts)  # keeps the order in which the alerts first appeared
        quoted = [a if n == 1 else f"{a} ({n} times)" for a, n in counts.items()]
        more = len(quoted) - MAX_ALERTS_LISTED
        return quoted[:MAX_ALERTS_LISTED] + ([f"and {more} more alerts of this level"] if more > 0 else [])

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

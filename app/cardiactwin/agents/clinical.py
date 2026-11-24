"""Clinical Recommendation agent. Current version: a report with beat counts only, priority is always routine.

Runs every window (also rejected ones). `report()` and `explain()` are called once at the end of a record.
"""
from collections import Counter
from typing import Literal, TypedDict, override

from .state import Agent, Label, PatientState

type Priority = Literal["routine", "review", "urgent"]


class Report(TypedDict):
    record: int
    minutes: float
    beats: dict[Label, int]        # number of beats per label
    priority: Priority


class ClinicalAgent(Agent):
    name = "clinical"

    @override
    def step(self, state: PatientState) -> None:
        pass

    def report(self, state: PatientState) -> Report:
        """Structured report: only facts taken from the state."""
        return Report(
            record=state.record,
            minutes=round(state.t / 60, 1),
            beats=dict(Counter(row["label"] for row in state.beat_log)),
            priority="routine",
        )

    def explain(self, report: Report) -> str:
        """Readable text for the clinician, from a fixed template."""
        beats = ", ".join(f"{k}: {v}" for k, v in sorted(report["beats"].items()))
        return f"Record {report['record']}, {report['minutes']} min. Beats: {beats}. Priority: {report['priority']}."

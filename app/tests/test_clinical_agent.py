"""Tests of the Clinical Recommendation agent on hand-made windows (no downloaded data needed).

usage (from the app/ folder):  pytest tests
"""
import sys
from pathlib import Path
from typing import override

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin.agents import Agent, PatientState, step_window  # noqa: E402
from cardiactwin.agents.clinical import ClinicalAgent  # noqa: E402
from cardiactwin.agents.state import Label  # noqa: E402
from test_agents_contract import RejectAll, make_window  # noqa: E402


class FixedLabels(Agent):
    """Stands in for Detection: gives the beats of each window the labels it is told."""
    name = "detection"

    def __init__(self, labels: list[Label]) -> None:
        self.labels = labels

    @override
    def step(self, state: PatientState) -> None:
        state.labels = self.labels[:len(state.window.r_peaks)]  # type: ignore[union-attr]
        state.probs = np.zeros((len(state.labels), 4))


def test_counts_beats_and_ventricular_burden() -> None:
    clinical, state = ClinicalAgent(), PatientState(0)
    labels: list[Label] = ["N"] * 8 + ["V"] * 2 + ["?"] * 2  # 12 beats
    step_window(state, make_window(t_start=10), [FixedLabels(labels), clinical])
    report = clinical.report(state)
    assert report["beats"] == {"N": 8, "V": 2, "?": 2}
    assert report["v_burden_pct"] == 20.0    # 2 of the 10 labeled beats
    assert report["unsure_pct"] == round(100 * 2 / 12, 1)


def test_rejected_windows_are_counted_and_not_in_the_burden() -> None:
    clinical, state = ClinicalAgent(), PatientState(0)
    step_window(state, make_window(t_start=10), [FixedLabels(["N"] * 12), clinical])
    step_window(state, make_window(t_start=20), [RejectAll(), FixedLabels(["V"] * 12), clinical])
    report = clinical.report(state)
    assert report["signal_rejected_pct"] == 50.0
    assert report["beats"] == {"N": 12, "-": 12} and report["v_burden_pct"] == 0.0


def test_heart_rate_from_usable_windows() -> None:
    clinical, state = ClinicalAgent(), PatientState(0)
    step_window(state, make_window(t_start=10, rr=0.8), [FixedLabels(["N"] * 12), clinical])
    step_window(state, make_window(t_start=20, n_beats=10, rr=1.0), [FixedLabels(["N"] * 10), clinical])
    hr = clinical.report(state)["heart_rate"]
    assert (hr["low"], hr["high"]) == (60, 75)


def test_alerts_and_twin_refits_are_counted() -> None:
    clinical, state = ClinicalAgent(), PatientState(0)
    step_window(state, make_window(t_start=10), [FixedLabels(["N"] * 12), clinical])
    state.alert("twin", "review", "the patient's normal beat has changed")
    state.alert("detection", "urgent", "run of 3 ventricular beats")
    state.decide("twin", "recalibrate", "drift")
    report = clinical.report(state)
    assert report["alerts"] == {"review": 1, "urgent": 1} and report["twin_refits"] == 1


def test_explain_mentions_the_main_facts() -> None:
    clinical, state = ClinicalAgent(), PatientState(7)
    step_window(state, make_window(t_start=10), [FixedLabels(["N"] * 12), clinical])
    text = clinical.explain(clinical.report(state))
    assert "Record 7" in text and "Priority: routine" in text and "75 bpm" in text

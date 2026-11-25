"""Tests of the Clinical Recommendation agent on hand-made windows (no downloaded data needed).

usage (from the app/ folder):  pytest tests
"""
import sys
from pathlib import Path
from typing import cast, override

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin.agents import Agent, PatientState, step_window  # noqa: E402
from cardiactwin.agents import llm  # noqa: E402
from cardiactwin.agents.clinical import SUMMARY_LABEL, ClinicalAgent, Report  # noqa: E402
from cardiactwin.agents.state import AlertLevel, Label  # noqa: E402
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
    labels = cast(list[Label], ["N"] * 8 + ["V"] * 2 + ["?"] * 2)  # 12 beats
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
    clinical, state = ClinicalAgent(llm=None), PatientState(7)
    step_window(state, make_window(t_start=10), [FixedLabels(["N"] * 12), clinical])
    text = clinical.explain(clinical.report(state))
    assert "Record 7" in text and "Priority: routine" in text and "75 bpm" in text


# --- priority -------------------------------------------------------------------------------------------


def report_for(labels: list[str], alerts: list[tuple[str, AlertLevel, str]] | None = None,
               rejected: bool = False) -> tuple[Report, PatientState]:
    """One usable window with these labels (plus one rejected window if asked), these alerts, then the report."""
    clinical, state = ClinicalAgent(), PatientState(0)
    step_window(state, make_window(t_start=10), [FixedLabels(cast(list[Label], labels)), clinical])
    if rejected:
        step_window(state, make_window(t_start=20), [RejectAll(), clinical])
    for agent, level, text in alerts or []:
        state.alert(agent, level, text)
    return clinical.report(state), state


def test_routine_when_nothing_stands_out() -> None:
    report, state = report_for(["N"] * 12)
    assert report["priority"] == "routine"
    assert state.log[-1]["agent"] == "clinical" and state.log[-1]["action"] == "routine"


def test_urgent_alert_makes_the_report_urgent_and_wins_over_review() -> None:
    report, state = report_for(["N"] * 6 + ["V"] * 6, [("detection", "urgent", "run of 3 ventricular beats"),
                                                       ("twin", "review", "normal beat changed")])
    assert report["priority"] == "urgent"
    assert report["reasons"] == ["detection: run of 3 ventricular beats"]
    assert state.log[-1]["action"] == "urgent"


def test_review_reasons() -> None:
    assert report_for(["N"] * 12, [("twin", "review", "normal beat changed")])[0]["priority"] == "review"
    burden, _ = report_for(["N"] * 10 + ["V"] * 2)  # 16.7% ventricular
    assert burden["priority"] == "review" and "ventricular" in burden["reasons"][0]
    unsure, _ = report_for(["N"] * 10 + ["?"] * 2)
    assert unsure["priority"] == "review" and "not sure" in unsure["reasons"][0]
    rejected, _ = report_for(["N"] * 12, rejected=True)  # 1 of 2 windows rejected
    assert rejected["priority"] == "review" and "rejected" in rejected["reasons"][0]


def test_only_the_first_alerts_are_quoted() -> None:
    report, _ = report_for(["N"] * 12, [("detection", "urgent", f"event {i}") for i in range(5)])
    assert report["reasons"] == ["detection: event 0", "detection: event 1", "detection: event 2",
                                 "and 2 more alerts of this level"]


def test_repeated_alerts_are_quoted_once_with_a_count() -> None:
    report, _ = report_for(["N"] * 12, [("twin", "review", "normal beat changed")] * 3)
    assert report["reasons"] == ["twin: normal beat changed (this alert was raised 3 times)"]


# --- text: template and optional LLM ---------------------------------------------------------------------


def explained(answer: str | Exception) -> tuple[str, str, PatientState]:
    """Explain a routine report (75 bpm) with a fake LLM; returns (text, template text, state)."""
    report, state = report_for(["N"] * 12)

    def fake_llm(instructions: str, text: str) -> str:
        if isinstance(answer, Exception):
            raise answer
        return answer

    template = ClinicalAgent(llm=None).explain(report)
    return ClinicalAgent(llm=fake_llm).explain(report, state), template, state


def text_decision(state: PatientState) -> str:
    return [row["reason"] for row in state.log if row["agent"] == "clinical" and row["action"] == "text"][-1]


def test_llm_summary_is_shown_above_the_facts_when_it_passes_the_checks() -> None:
    answer = "Priority: routine. All beats were normal and the heart rate was 75 bpm."
    text, template, state = explained(answer)
    assert text == f"{SUMMARY_LABEL}\n{answer}\n\nFacts:\n{template}"
    assert text_decision(state).startswith("LLM summary added")


def test_llm_answer_with_an_invented_number_is_left_out() -> None:
    text, template, state = explained("Priority: routine. The heart rate was 80 bpm.")
    assert text == template and "80" in text_decision(state)


def test_llm_answer_without_the_priority_is_left_out() -> None:
    text, template, state = explained("The heart rate was 75 bpm.")
    assert text == template and "priority" in text_decision(state)


def test_llm_failure_leaves_only_the_facts() -> None:
    text, template, state = explained(ConnectionError("no server"))
    assert text == template and "ConnectionError" in text_decision(state)


def test_llm_is_off_unless_the_environment_names_a_model(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv(llm.MODEL_ENV, raising=False)
    assert ClinicalAgent().llm is None
    monkeypatch.setenv(llm.MODEL_ENV, "some-model")
    assert callable(ClinicalAgent().llm)


def test_numbers_are_compared_by_value() -> None:
    text, _, state = explained("Priority: routine. 0% of the signal was rejected; heart rate 75.0 bpm.")
    assert text.startswith(SUMMARY_LABEL) and text_decision(state).startswith("LLM summary added")

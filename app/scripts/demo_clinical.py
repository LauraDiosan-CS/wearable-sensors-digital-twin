"""Show how the Clinical Recommendation agent reacts in three made-up situations: routine, review and urgent.

The other agents' outputs (beat labels, alerts) are simulated here, so the demo works before the real Signal
Monitoring, Detection and Prediction agents exist. No downloaded data is needed. If CARDIACTWIN_OLLAMA_MODEL is
set, the text comes from the local LLM (see cardiactwin/agents/llm.py).

usage: python scripts/demo_clinical.py   (run from the app/ folder)
"""
import sys
from pathlib import Path
from typing import override

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import data  # noqa: E402
from cardiactwin.agents import Agent, PatientState, Window, step_window  # noqa: E402
from cardiactwin.agents.clinical import ClinicalAgent  # noqa: E402
from cardiactwin.agents.state import Label  # noqa: E402

RR = 0.8          # 75 bpm
N_WINDOWS = 30    # 5 minutes


def window(k: int) -> Window:
    """10 s with a beat every RR seconds (the signal itself does not matter for this demo)."""
    n = int(10 / RR)
    r_peaks = data.PRE + (np.arange(n) * RR * data.FS).astype(np.int64)
    return Window(record=0, t_start=10.0 * k, t_end=10.0 * k + 10, start=int(10 * k * data.FS) - data.PRE,
                  signal=np.zeros(10 * data.FS + data.PRE + data.POST), r_peaks=r_peaks, rr=np.full((n, 3), RR))


class Simulated(Agent):
    """Stands in for the other agents: gives each window's beats labels and raises the alerts of a scenario."""
    name = "detection"

    def __init__(self, labels: list[Label], alerts: dict[int, tuple[str, str, str]],
                 windows: dict[int, list[Label]] | None = None) -> None:
        self.labels, self.alerts, self.windows, self.k = labels, alerts, windows or {}, 0

    @override
    def step(self, state: PatientState) -> None:
        n = len(state.window.r_peaks)  # type: ignore[union-attr]
        state.labels = self.windows.get(self.k) or [self.labels[(self.k * n + i) % len(self.labels)] for i in range(n)]
        state.probs = np.zeros((n, 4))
        if self.k in self.alerts:
            agent, level, text = self.alerts[self.k]
            state.alert(agent, level, text)  # type: ignore[arg-type]
        self.k += 1


SCENARIOS: dict[str, tuple[list[Label], dict[int, tuple[str, str, str]], dict[int, list[Label]]]] = {
    "routine: only normal beats, no alerts": (["N"], {}, {}),
    "review: many ventricular and unclear beats, and the twin saw the normal beat change": (
        ["N", "N", "V", "N", "?", "N", "N", "V", "N", "?", "N", "N"],
        {12: ("twin", "review", "the patient's normal beat has changed; the twin was recalibrated")}, {}),
    "urgent: a run of ventricular beats": (
        ["N"],
        {20: ("detection", "urgent", "run of 4 ventricular beats at 3:20")},
        {20: ["N", "N", "N", "N", "V", "V", "V", "V", "N", "N", "N", "N"]}),  # the alert's window holds the 4 V beats
}


def main() -> None:
    for title, (labels, alerts, windows) in SCENARIOS.items():
        clinical, state = ClinicalAgent(), PatientState(record=0)
        agents = [Simulated(labels, alerts, windows), clinical]
        for k in range(N_WINDOWS):
            step_window(state, window(k), agents)
        report = clinical.report(state)
        text = clinical.explain(report, state)
        print(f"\n=== {title}")
        print(text)
        print("decisions of the Clinical agent:")
        for row in state.log:
            if row["agent"] == "clinical":
                print(f"  {row['action']:<8} {row['reason']}")


if __name__ == "__main__":
    main()

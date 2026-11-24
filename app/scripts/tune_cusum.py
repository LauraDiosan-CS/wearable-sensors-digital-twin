"""Tune the Twin agent's drift CUSUM (CUSUM_K, CUSUM_H) on the training patients (DS1).

The Twin agent runs exactly as in the system: when its CUSUM fires, it refits. For each setting we measure
- refits per hour: how restless the twin is (fewer is calmer);
- normal beats scoring above 3: normal beats that still look abnormal, i.e. false alarms the twin would cause.
A good setting keeps both low. "none" (never refit) shows what happens without any updates.

Detection is replaced here by the cardiologists' labels (a perfect detector). That is allowed on the training
patients; the chosen values should be re-checked once the real Detection agent exists.

usage: python scripts/tune_cusum.py   (run from the app/ folder)
"""
import sys
from pathlib import Path
from typing import override

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import data  # noqa: E402
from cardiactwin.agents import Agent, PatientState, step_window  # noqa: E402
from cardiactwin.agents.orchestrator import Truth, replay  # noqa: E402
from cardiactwin.agents.signal_monitoring import SignalMonitoringAgent  # noqa: E402
from cardiactwin.agents.state import Label  # noqa: E402
from cardiactwin.agents.twin_agent import Policy, TwinAgent  # noqa: E402
from cardiactwin.style import save_json  # noqa: E402

HIGH_SCORE = 3.0
SETTINGS: list[tuple[Policy, float, float]] = [  # (policy, K, H)
    ("none", 0, 0),
    ("drift", 0.5, 20),
    ("drift", 0.5, 40),
    ("drift", 1.0, 20),
    ("drift", 1.0, 40),
]


class OracleDetection(Agent):
    """Tuning only: copies the cardiologists' labels of the current window, all with probability 1."""
    name = "detection"

    def __init__(self) -> None:
        self.truth: Truth = np.array([])

    @override
    def step(self, state: PatientState) -> None:
        labels: list[Label] = [x if x in ("N", "S", "V", "F") else "?" for x in self.truth]  # type: ignore[misc]
        probs = np.zeros((len(labels), 4))
        for i, x in enumerate(labels):
            if x != "?":
                probs[i, ("N", "S", "V", "F").index(x)] = 1.0
        state.labels, state.probs = labels, probs


def run(record: int, policy: Policy, k: float, h: float) -> tuple[int, float, list[float]]:
    """Run Signal + Twin + perfect detection; return refits, hours after calibration, normal-beat scores."""
    oracle = OracleDetection()
    state = PatientState(record)
    agents = [SignalMonitoringAgent(), TwinAgent(policy, cusum_k=k, cusum_h=h), oracle]
    for window, truth in replay(record):
        oracle.truth = truth
        step_window(state, window, agents)
    refits = sum(1 for row in state.log if row["agent"] == "twin" and row["action"] == "recalibrate")
    scored = [b for b in state.beat_log if b["deviation"] is not None]
    hours = (scored[-1]["t"] - scored[0]["t"]) / 3600 if scored else 0.0
    return refits, hours, [b["deviation"] for b in scored if b["label"] == "N"]  # type: ignore[misc]


def main() -> None:
    results = []
    for policy, k, h in SETTINGS:
        name = policy if policy == "none" else f"drift K={k:g} H={h:g}"
        refits, hours = 0, 0.0
        scores: list[float] = []
        for record in data.DS1:
            r, hr, s = run(record, policy, k, h)
            refits, hours, scores = refits + r, hours + hr, scores + s
        high = 100 * float(np.mean(np.array(scores) > HIGH_SCORE))
        results.append({"setting": name, "policy": policy, "K": k, "H": h, "refits_per_hour": refits / hours,
                        "normal_above_3_pct": high, "normal_median": float(np.median(scores))})
        print(f"{name:<18} refits/hour {refits / hours:5.2f}   normal beats > {HIGH_SCORE:g}: {high:5.2f}%   "
              f"median normal score {np.median(scores):.2f}", flush=True)
    save_json({"records": data.DS1, "settings": results}, "cusum_tuning")


if __name__ == "__main__":
    main()

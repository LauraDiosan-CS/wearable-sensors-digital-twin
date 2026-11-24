"""Tune the Twin agent's drift CUSUM (CUSUM_K, CUSUM_H) on the training patients (DS1).

DS1 recordings have no known drift, so every CUSUM alarm on them counts as a false alarm. We look for values
that (almost) never fire on these stable recordings but still react quickly to a real change.

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
from cardiactwin.agents.state import FloatArray, Label  # noqa: E402
from cardiactwin.agents.twin_agent import TwinAgent  # noqa: E402
from cardiactwin.style import save_json  # noqa: E402

K_GRID = (0.25, 0.5, 0.75, 1.0, 1.5)
H_GRID = (5, 10, 20, 40, 80)
SHIFTED_SCORE = 3.0  # for the reaction time: how fast would beats that all score 3 trigger an alarm?


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


def normal_scores(record: int) -> tuple[FloatArray, float]:
    """Deviation scores of the normal beats after calibration, in time order, and the hours they cover."""
    oracle = OracleDetection()
    state, agents = PatientState(record), [SignalMonitoringAgent(), TwinAgent("none"), oracle]
    for window, truth in replay(record):
        oracle.truth = truth
        step_window(state, window, agents)
    rows = [b for b in state.beat_log if b["deviation"] is not None and b["label"] == "N"]
    hours = (rows[-1]["t"] - rows[0]["t"]) / 3600 if rows else 0.0
    return np.array([b["deviation"] for b in rows]), hours


def alarms(scores: FloatArray, k: float, h: float) -> int:
    """How often the CUSUM passes h (it restarts from 0 after each alarm, like after a refit)."""
    cusum, n = 0.0, 0
    for d in scores:
        cusum = max(0.0, cusum + d - 1 - k)
        if cusum > h:
            n, cusum = n + 1, 0.0
    return n


def main() -> None:
    per_record = {}
    for i, record in enumerate(data.DS1):
        per_record[record] = normal_scores(record)
        print(f"[{i + 1}/{len(data.DS1)}] record {record}: {len(per_record[record][0])} normal beats", flush=True)
    hours = sum(h for _, h in per_record.values())
    grid = []
    print(f"\n{hours:.1f} hours of normal beats. False alarms per hour (records with any alarm), "
          f"and beats needed to react if normal beats suddenly score {SHIFTED_SCORE:g}:")
    print("  K \\ H " + "".join(f"{h:>16g}" for h in H_GRID))
    for k in K_GRID:
        cells = []
        for h in H_GRID:
            counts = {r: alarms(s, k, h) for r, (s, _) in per_record.items()}
            total = sum(counts.values())
            react = int(np.ceil(h / (SHIFTED_SCORE - 1 - k)))
            grid.append({"K": k, "H": h, "false_alarms": total, "per_hour": total / hours,
                         "records_with_alarm": [r for r, c in counts.items() if c], "react_beats": react})
            cells.append(f"{total / hours:5.2f} ({sum(1 for c in counts.values() if c):2d}) {react:3d}b")
        print(f"  {k:5g} " + "".join(f"{c:>16}" for c in cells))
    save_json({"hours": hours, "grid": grid}, "cusum_tuning")


if __name__ == "__main__":
    main()

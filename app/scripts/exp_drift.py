"""Experiment E4: does recalibration keep the twin useful when the patient's normal beat drifts?

MIT-BIH has no controlled drift, so we add one to the test patients (DS2): from minute 10 to minute 25 the
T wave of every beat (0.2-0.45 s after the R peak) turns over linearly until it is inverted to -50% of its
original height (T-wave inversion), then stays that way. The same recordings are run with the Twin agent's
three policies (none, scheduled, drift), once with this drift and once without it (control): the control
shows what the policies do on the natural changes real recordings already contain.

Detection is replaced by the cardiologists' labels (a perfect detector), so the experiment measures the twin's
policy alone. Rerun with the real Detection agent once it exists.

usage: python scripts/exp_drift.py   (run from the app/ folder)
"""
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import data  # noqa: E402
from cardiactwin.agents import PatientState, Window, step_window  # noqa: E402
from cardiactwin.agents.orchestrator import replay  # noqa: E402
from cardiactwin.agents.signal_monitoring import SignalMonitoringAgent  # noqa: E402
from cardiactwin.agents.twin_agent import Policy, TwinAgent  # noqa: E402
from cardiactwin.style import plt, save_fig, save_json  # noqa: E402
from tune_cusum import OracleDetection  # noqa: E402

DRIFT_START_S, DRIFT_END_S = 600.0, 1500.0  # minute 10 to minute 25
T_FACTOR_END = -0.5                         # the T wave ends inverted, at -50% of its height
T_FROM_S, T_TO_S = 0.20, 0.45               # where the T wave is, relative to the R peak
HIGH_SCORE = 3.0
POLICIES: tuple[Policy, ...] = ("none", "scheduled", "drift")
EXAMPLE_RECORD = 113                        # a patient with a clearly visible T wave


def t_factor(t: float) -> float:
    """Scale of the T wave at time t: 1 = untouched, -0.5 = inverted to half height."""
    progress = np.clip((t - DRIFT_START_S) / (DRIFT_END_S - DRIFT_START_S), 0, 1)
    return float(1 - (1 - T_FACTOR_END) * progress)


def inject_drift(window: Window) -> Window:
    """A copy of the window in which every beat's T wave is scaled by t_factor (smoothly, with no steps)."""
    signal = window.signal.copy()
    a, b = int(T_FROM_S * data.FS), int(T_TO_S * data.FS)
    taper = np.hanning(b - a)  # 0 at the edges, 1 in the middle
    for r, t in zip(window.r_peaks, window.beat_times):
        seg = signal[r + a:r + b]
        baseline = np.linspace(seg[0], seg[-1], len(seg))  # straight line under the T wave
        seg -= (1 - t_factor(t)) * taper * (seg - baseline)
    return dataclasses.replace(window, signal=signal)


def run(record: int, policy: Policy, drift: bool) -> tuple[PatientState, TwinAgent]:
    oracle, twin_agent = OracleDetection(), TwinAgent(policy)
    state, agents = PatientState(record), [SignalMonitoringAgent(), twin_agent, oracle]
    for window, truth in replay(record):
        oracle.truth = truth
        step_window(state, inject_drift(window) if drift else window, agents)
    return state, twin_agent


def normal_scores(state: PatientState, t_from: float, t_to: float) -> list[float]:
    return [b["deviation"] for b in state.beat_log  # type: ignore[misc]
            if b["label"] == "N" and b["deviation"] is not None and t_from <= b["t"] < t_to]


def main() -> None:
    results: dict[str, dict] = {}
    example: dict[Policy, tuple[PatientState, list[float]]] = {}
    for drift in (False, True):
        for policy in POLICIES:
            name = f"{'drift' if drift else 'control'} / {policy}"
            before: list[float] = []
            after: list[float] = []
            end: list[float] = []
            delays: list[float] = []
            refits = alerts = 0
            for i, record in enumerate(data.DS2):
                state, twin_agent = run(record, policy, drift)
                before += normal_scores(state, 300, DRIFT_START_S)
                after += normal_scores(state, DRIFT_START_S, np.inf)
                end += normal_scores(state, DRIFT_END_S, np.inf)
                fits = [f["t"] for f in twin_agent.history[1:]]  # refits, without the first calibration
                refits += len(fits)
                alerts += len(state.alerts)
                if policy != "none" and any(t >= DRIFT_START_S for t in fits):
                    delays.append(min(t for t in fits if t >= DRIFT_START_S) - DRIFT_START_S)
                if drift and record == EXAMPLE_RECORD:
                    example[policy] = (state, fits)
                print(f"[{name} {i + 1}/{len(data.DS2)}] record {record}: {len(fits)} refits", flush=True)
            results[name] = {
                "normal_median_before": float(np.median(before)),
                "normal_median_end": float(np.median(end)),
                "normal_above_3_pct_after_start": 100 * float(np.mean(np.array(after) > HIGH_SCORE)),
                "refits_per_record": refits / len(data.DS2),
                "alerts_per_record": alerts / len(data.DS2),
                "first_refit_after_drift_start_s": float(np.median(delays)) if delays else None,
            }
    print(f"\n{'':<20}{'median before':>14}{'median end':>12}{'>3 after min 10':>17}{'refits':>8}{'alerts':>8}"
          f"{'1st refit after min 10':>24}")
    for name, r in results.items():
        delay = "-" if r["first_refit_after_drift_start_s"] is None else f"{r['first_refit_after_drift_start_s']:.0f} s"
        print(f"{name:<20}{r['normal_median_before']:>14.2f}{r['normal_median_end']:>12.2f}"
              f"{r['normal_above_3_pct_after_start']:>16.2f}%{r['refits_per_record']:>8.2f}"
              f"{r['alerts_per_record']:>8.2f}{delay:>24}")
    save_json({"records": data.DS2, "drift": {"start_s": DRIFT_START_S, "end_s": DRIFT_END_S,
                                              "t_factor_end": T_FACTOR_END}, "results": results}, "drift")
    plot_example(example)


def plot_example(example: dict[Policy, tuple[PatientState, list[float]]]) -> None:
    """Normal-beat score over time for one record under the three policies, with the refits marked."""
    colors = {"none": "#7f7f7f", "scheduled": "#c77c02", "drift": "#1f4e79"}
    fig, ax = plt.subplots(figsize=(6.7, 2.4))
    ax.axvspan(DRIFT_START_S / 60, DRIFT_END_S / 60, color="#000000", alpha=0.06, lw=0)
    for policy, (state, fits) in example.items():
        rows = [b for b in state.beat_log if b["label"] == "N" and b["deviation"] is not None]
        t = np.array([b["t"] for b in rows])
        d = np.array([b["deviation"] for b in rows], dtype=float)
        edges = np.arange(300, t.max() + 30, 30)
        idx = np.digitize(t, edges)
        mids = [(edges[i - 1] + 15) / 60 for i in np.unique(idx) if 0 < i < len(edges)]
        meds = [np.median(d[idx == i]) for i in np.unique(idx) if 0 < i < len(edges)]
        ax.plot(mids, meds, color=colors[policy], lw=1.3, label=policy)
        for f in fits:
            ax.axvline(f / 60, color=colors[policy], lw=0.6, ls=":")
    ax.set_xlabel("minutes")
    ax.set_ylabel("median score of normal beats (30 s)")
    ax.set_title(f"record {EXAMPLE_RECORD}: T wave inverted between minute 10 and 25 (shaded); dotted = refits")
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, "s6_drift")


if __name__ == "__main__":
    main()

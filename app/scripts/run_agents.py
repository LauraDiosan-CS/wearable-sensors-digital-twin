"""Run the multi-agent system on test records, save each clinical report and print one example.

Prints one line per record (priority, twin refits, alerts), then the first record in full: the clinician's text
and the decisions of the Twin and Clinical agents. Ends with how many records got each priority and the
beat-level F1 against the cardiologists' labels.

usage: python scripts/run_agents.py [record ...]   (run from the app/ folder; default: all DS2 records)
"""
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from cardiactwin import classify as cl, data  # noqa: E402
from cardiactwin.agents import run_record  # noqa: E402
from cardiactwin.agents.clinical import ClinicalAgent  # noqa: E402
from cardiactwin.style import OUT_DIR, save_json  # noqa: E402


def to_index(labels: Iterable[str]) -> NDArray[np.int64]:
    """Class letters to indices; "?" (abstained) and "-" (rejected) become -1, i.e. never a correct label."""
    return np.array([data.CLASSES.index(x) if x in data.CLASSES else -1 for x in labels], dtype=np.int64)


def main() -> None:
    records = [int(r) for r in sys.argv[1:]] or data.DS2
    out_dir = OUT_DIR / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true: list[NDArray[np.int64]] = []
    y_pred: list[NDArray[np.int64]] = []
    priorities: dict[int, str] = {}
    example = ""
    print(f"{'record':>6}  {'priority':<8}  {'twin refits':>11}  {'alerts':>6}  first reason")
    for i, record in enumerate(records):
        state, agents, truth = run_record(record)
        clinical = next(a for a in agents if isinstance(a, ClinicalAgent))
        report = clinical.report(state)
        text = clinical.explain(report, state)
        with open(out_dir / f"{record}.json", "w", encoding="utf-8") as f:
            json.dump({"report": report, "text": text, "alerts": state.alerts, "log": state.log}, f, indent=1)
        y_true.append(to_index(truth))
        y_pred.append(to_index(row["label"] for row in state.beat_log))
        priorities[record] = report["priority"]
        print(f"{record:>6}  {report['priority']:<8}  {report['twin_refits']:>11}  {len(state.alerts):>6}  "
              f"{report['reasons'][0]}", flush=True)
        if i == 0:
            decisions = [row for row in state.log if row["agent"] in ("twin", "clinical")]
            example = (f"\n=== example: record {record}\n{text}\n\ndecisions of the Twin and Clinical agents:\n"
                       + "\n".join(f"  t={row['t']:6.0f} s  {row['agent']:<8}  {row['action']:<11}  {row['reason']}"
                                   for row in decisions))
    print(example)
    counts = Counter(priorities.values())
    print(f"\npriorities: {', '.join(f'{p} {counts[p]}' for p in ('routine', 'review', 'urgent'))}")
    m = cl.metrics(np.concatenate(y_true), np.concatenate(y_pred))
    summary = {"records": records, "priorities": priorities, "macro_F1": m["macro_F1"],
               **{c: m[c] for c in cl.EVAL_CLASSES}}
    save_json(summary, "agents")
    print("beat-level macro-F1", round(m["macro_F1"], 3), {c: round(m[c]["F1"], 3) for c in cl.EVAL_CLASSES})


if __name__ == "__main__":
    main()

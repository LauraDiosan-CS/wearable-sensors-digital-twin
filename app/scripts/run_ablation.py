"""Key experiment: does the twin's deviation score help the beat classifier on unseen patients?

The classifier is deterministic, so uncertainty comes from the patients: differences are reported
with 95% intervals from a bootstrap over the test patients (each patient is one independent unit).

usage: python scripts/run_ablation.py   (after run_twin.py; run from the app/ folder)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import classify as cl, data  # noqa: E402
from cardiactwin.style import OUT_DIR, save_json  # noqa: E402

BOOTSTRAP = 2000


def deviation(records, key, dev):
    return np.concatenate([dev[f"{r}_{key}"] for r in records])


def summary(m):
    return {"macro_F1": m["macro_F1"], **{c: m[c] for c in cl.EVAL_CLASSES}}


def bootstrap_diff(y, preds, base, other, records, rng):
    """95% interval of (other - base) for macro-F1 and per-class F1, resampling test patients."""
    uniq = np.unique(records)
    idx = {r: np.flatnonzero(records == r) for r in uniq}
    keys = ["macro_F1"] + [f"{c}_F1" for c in cl.EVAL_CLASSES]
    diffs = {k: [] for k in keys}
    for _ in range(BOOTSTRAP):
        sel = np.concatenate([idx[r] for r in rng.choice(uniq, size=len(uniq), replace=True)])
        mb, mo = cl.metrics(y[sel], preds[base][sel]), cl.metrics(y[sel], preds[other][sel])
        diffs["macro_F1"].append(mo["macro_F1"] - mb["macro_F1"])
        for c in cl.EVAL_CLASSES:
            diffs[f"{c}_F1"].append(mo[c]["F1"] - mb[c]["F1"])
    return {k: {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
            for k, v in diffs.items()}


def main():
    dev = dict(np.load(OUT_DIR / "twin_deviation.npz"))
    train_all, test_all = data.load_split(data.DS1), data.load_split(data.DS2)
    mtr, mte = cl.after_calibration(train_all), cl.after_calibration(test_all)
    train, test = cl.subset(train_all, mtr), cl.subset(test_all, mte)

    variants = {
        "without deviation": (None, None),
        "with twin deviation": (deviation(data.DS1, "d_twin", dev)[mtr], deviation(data.DS2, "d_twin", dev)[mte]),
        "with template deviation (control)": (deviation(data.DS1, "d_template", dev)[mtr],
                                              deviation(data.DS2, "d_template", dev)[mte]),
    }
    results = {"train_beats": int(len(train["labels"])), "test_beats": int(len(test["labels"])),
               "test_patients": int(len(np.unique(test["record"]))), "bootstrap": BOOTSTRAP}
    preds = {}
    for name, (dtr, dte) in variants.items():
        ftr = cl.features(train, None if dtr is None else np.log(dtr))
        fte = cl.features(test, None if dte is None else np.log(dte))
        preds[name] = cl.train_baseline(ftr, train["labels"]).predict(fte)
        results[name] = summary(cl.metrics(test["labels"], preds[name]))
        print(name, round(results[name]["macro_F1"], 4),
              {c: round(results[name][c]["F1"], 3) for c in cl.EVAL_CLASSES}, flush=True)

    rng = np.random.default_rng(0)
    for other in ("with twin deviation", "with template deviation (control)"):
        results[f"difference: {other}"] = bootstrap_diff(test["labels"], preds, "without deviation", other,
                                                         test["record"], rng)
        print(other, {k: (round(v["mean"], 3), [round(x, 3) for x in v["ci95"]])
                      for k, v in results[f"difference: {other}"].items()}, flush=True)
    save_json(results, "ablation")


if __name__ == "__main__":
    main()

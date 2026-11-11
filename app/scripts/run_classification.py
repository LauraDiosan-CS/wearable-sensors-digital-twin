"""Train the beat classifiers on DS1 (training patients) and test them on DS2 (unseen patients).

usage: python scripts/run_classification.py   (run from the app/ folder)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import classify as cl, data  # noqa: E402
from cardiactwin.style import save_json  # noqa: E402

SEEDS = (0, 1, 2)


def main():
    train = data.load_split(data.DS1)
    ds2 = data.load_split(data.DS2)
    test = cl.subset(ds2, cl.after_calibration(ds2))  # same test beats as the twin experiments
    print(f"train beats {len(train['labels'])}, test beats {len(test['labels'])}", flush=True)

    results = {"test_beats": int(len(test["labels"]))}
    gbm = cl.train_baseline(cl.features(train), train["labels"])
    results["baseline"] = cl.metrics(test["labels"], gbm.predict(cl.features(test)))
    print("baseline", results["baseline"]["macro_F1"], flush=True)

    runs = []
    for seed in SEEDS:
        model = cl.train_cnn(train, seed=seed, log=lambda m: print(m, flush=True))
        runs.append(cl.metrics(test["labels"], cl.predict_cnn(model, test)))
        print("cnn seed", seed, runs[-1]["macro_F1"], flush=True)
    results["cnn_runs"] = runs
    results["cnn_mean"] = {
        "macro_F1": float(np.mean([r["macro_F1"] for r in runs])),
        "macro_F1_std": float(np.std([r["macro_F1"] for r in runs])),
        **{c: {k: float(np.mean([r[c][k] for r in runs])) for k in ("Se", "+P", "F1")}
           for c in cl.EVAL_CLASSES},
    }
    save_json(results, "classification")
    print(results["cnn_mean"])


if __name__ == "__main__":
    main()

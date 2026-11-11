"""Forecast RR intervals: persistence vs a linear autoregressive model (train DS1, test DS2).

usage: python scripts/run_forecasting.py   (run from the app/ folder)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import data, forecast as fc  # noqa: E402
from cardiactwin.style import save_json  # noqa: E402


def mae_ms(y, p):
    return float(np.mean(np.abs(y - p)) * 1000)


def main():
    xtr, y1tr, yhtr = fc.windows(data.DS1)
    xte, y1te, yhte = fc.windows(data.DS2)
    results = {"train_windows": int(len(xtr)), "test_windows": int(len(xte))}
    for name, ytr, yte, h in (("next_beat", y1tr, y1te, 1), (f"next_{fc.HORIZON}_beats", yhtr, yhte, fc.HORIZON)):
        ar = fc.fit_ar(xtr, ytr)
        results[name] = {"persistence_mae_ms": mae_ms(yte, fc.persistence(xte, h)),
                         "autoregressive_mae_ms": mae_ms(yte, fc.predict_ar(ar, xte))}
        print(name, results[name], flush=True)
    save_json(results, "forecasting")


if __name__ == "__main__":
    main()

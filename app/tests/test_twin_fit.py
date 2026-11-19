"""Synthetic check of twin calibration: does the fit recover a known beat? (no data needed)

For 3 heart rates x 3 random parameter sets: simulate a beat from known parameters, scale it to mV,
add noise, fit the twin to the noisy beat, and compare the fitted beat with the clean one.
A test passes if the fit is closer to the clean beat than the added noise.

usage (from the app/ folder):  python tests/test_twin_fit.py   or   pytest tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import twin as tw  # noqa: E402

NOISE_MV = 0.02  # standard deviation of the added noise


def run_checks():
    """Returns (heart rate, test number, fit error in mV) for all nine tests."""
    rng = np.random.default_rng(0)
    results = []
    for hr in (60, 80, 100):
        rr = 60 / hr
        for k in range(3):
            true = tw.default_params(rr) * rng.uniform(0.8, 1.2, size=15)  # every parameter +-20%
            clean = tw.model_beat(true, rr)
            clean = clean / np.ptp(clean) * 1.2  # about 1.2 mV peak to peak, like a real beat
            noisy = clean + rng.normal(0, NOISE_MV, size=clean.shape)
            _, fitted = tw.fit(noisy, rr)
            results.append((hr, k + 1, float(tw.rms(fitted - clean))))
    return results


def test_twin_recovers_known_beat():
    for hr, k, err in run_checks():
        assert err < NOISE_MV, f"{hr} bpm, test {k}: error {err:.4f} mV"


if __name__ == "__main__":
    results = run_checks()
    for hr, k, err in results:
        print(f"{hr:3d} bpm, test {k}: error {err:.4f} mV  {'ok' if err < NOISE_MV else 'FAIL'}")
    print(f"{sum(err < NOISE_MV for *_, err in results)}/{len(results)} passed (noise {NOISE_MV} mV)")

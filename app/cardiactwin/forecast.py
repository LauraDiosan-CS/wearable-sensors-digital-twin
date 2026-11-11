"""Physiological Prediction agent, first version: forecasting the next RR intervals."""
import numpy as np
from sklearn.linear_model import Ridge

from . import data

WINDOW = 20  # past RR intervals used as input
HORIZON = 10  # forecast the average RR of the next 10 beats


def windows(records):
    """Past windows, the next RR and the mean of the next HORIZON RR intervals (seconds)."""
    xs, y1, yh = [], [], []
    for r in records:
        rr = data.extract_beats(r)["rr_all"].astype(np.float64)
        for k in range(WINDOW, len(rr) - HORIZON):
            xs.append(rr[k - WINDOW:k])
            y1.append(rr[k])
            yh.append(rr[k:k + HORIZON].mean())
    return np.asarray(xs), np.asarray(y1), np.asarray(yh)


def relative(x):
    """Express each window relative to its own mean so one model fits every heart rate."""
    m = x.mean(axis=1, keepdims=True)
    return x / m, m[:, 0]


def fit_ar(x, y):
    xr, m = relative(x)
    return Ridge(alpha=1.0).fit(xr, y / m)


def predict_ar(model, x):
    xr, m = relative(x)
    return model.predict(xr) * m


def persistence(x, horizon):
    return x[:, -1] if horizon == 1 else x[:, -HORIZON:].mean(axis=1)

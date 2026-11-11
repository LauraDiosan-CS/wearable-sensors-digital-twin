"""Twin Calibration agent, first version: per-patient fit of the McSharry et al. (2003) ECG model.

The model's z coordinate obeys dz/dt = -sum_i a_i dth_i exp(-dth_i^2 / (2 b_i^2)) - z, with the phase
theta advancing at 2*pi/RR. For a constant RR this is a linear system driven by a periodic input, so
the periodic steady state can be computed exactly from one simulated cycle (no start-up transient).
Default parameters are those of the PhysioNet ECGSYN code.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter

from . import data

TI_DEG = np.array([-70.0, -15.0, 0.0, 15.0, 100.0])  # P, Q, R, S, T angles (degrees)
AI = np.array([1.2, -5.0, 30.0, -7.5, 0.75])
BI = np.array([0.25, 0.1, 0.1, 0.1, 0.4])
UPSAMPLE = 4
WAVES = ["P", "Q", "R", "S", "T"]


def default_params(rr):
    """ECGSYN defaults, with angles and widths rescaled for the heart rate as in ECGSYN."""
    hrfact = np.sqrt((60.0 / rr) / 60.0)
    hrfact2 = np.sqrt(hrfact)
    ti = np.array([hrfact2, hrfact, 1.0, hrfact, hrfact2]) * TI_DEG * np.pi / 180
    return np.concatenate([ti, AI, BI * hrfact])


def model_beat(params, rr, fs=data.FS, pre=data.PRE, post=data.POST):
    """Periodic model ECG (arbitrary units) in the same window around the R peak as the real beats."""
    ti, ai, bi = params[:5], params[5:10], params[10:15]
    fs_int = fs * UPSAMPLE
    m = int(round(rr * fs_int))
    theta = -np.pi + 2 * np.pi * np.arange(m) / m  # R wave (theta = 0) at sample m // 2
    dti = np.fmod(theta[:, None] - ti[None, :], 2 * np.pi)
    u = -np.sum(ai * dti * np.exp(-0.5 * (dti / bi) ** 2), axis=1)
    e = np.exp(-1.0 / fs_int)
    z = lfilter([1 - e], [1, -e], u)  # response from z = 0
    c = z[-1] / (1 - e ** m)  # initial state that makes the response periodic
    z = z + c * e ** (np.arange(m) + 1)
    idx = (m // 2 + np.arange(-pre * UPSAMPLE, post * UPSAMPLE, UPSAMPLE)) % m
    return z[idx]


def _affine(z, b):
    """Best gain and offset mapping model units to mV (closed-form least squares)."""
    a = np.stack([z, np.ones_like(z)], axis=1)
    coef, *_ = np.linalg.lstsq(a, b, rcond=None)
    return a @ coef, coef


def fit(beat, rr, lam=1e-3):
    """Calibrate the 15 wave parameters to one beat template; returns (params, fitted beat)."""
    p0 = default_params(rr)
    scale = np.concatenate([np.full(5, 0.2), np.abs(AI) * 0.5 + 0.1, p0[10:] * 0.5])
    lo = np.concatenate([p0[:5] - 0.5, np.where(AI > 0, 0.0, AI * 4), np.full(5, 0.01)])
    hi = np.concatenate([p0[:5] + 0.5, np.where(AI > 0, AI * 4, 0.0), p0[10:] * 4])
    norm = np.std(beat) + 1e-9

    def residual(p):
        g, _ = _affine(model_beat(p, rr), beat)
        return np.concatenate([(g - beat) / norm, np.sqrt(lam) * (p - p0) / scale])

    res = least_squares(residual, p0, bounds=(lo, hi), x_scale=scale, max_nfev=300)
    fitted, _ = _affine(model_beat(res.x, rr), beat)
    return res.x, fitted


def default_fit(beat, rr):
    """Population twin: default parameters, only gain and offset adapted."""
    fitted, _ = _affine(model_beat(default_params(rr), rr), beat)
    return fitted


def rms(x, axis=-1):
    return np.sqrt(np.mean(x ** 2, axis=axis))


def calibrate_record(d, seconds=data.CALIB_SECONDS):
    """Fit the twin to the median beat of the first `seconds` of a record (no labels used)."""
    calib = d["time_s"] < seconds
    template = np.median(d["beats"][calib], axis=0)
    rr = float(np.median(d["pre_rr"][calib]))
    params, twin = fit(template, rr)
    sigma_twin = float(np.median(rms(d["beats"][calib] - twin)))
    sigma_tmpl = float(np.median(rms(d["beats"][calib] - template)))
    return {
        "params": params, "rr": rr, "template": template, "twin": twin,
        "default": default_fit(template, rr),
        "sigma_twin": sigma_twin, "sigma_template": sigma_tmpl,
        # deviation of every beat from the patient's own twin, and from the raw median beat (control)
        "d_twin": rms(d["beats"] - twin) / sigma_twin,
        "d_template": rms(d["beats"] - template) / sigma_tmpl,
    }

"""Exploratory data analysis of MIT-BIH (inter-patient split) and the Noise Stress Test noise.

usage: python scripts/eda.py   (run from the app/ folder, after download_data.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cardiactwin import data  # noqa: E402
from cardiactwin.style import CLASS_COLORS, plt, save_fig, save_json  # noqa: E402


def class_counts(split):
    return {c: int((split["labels"] == i).sum()) for i, c in enumerate(data.CLASSES)}


def rms(x, axis=-1):
    return np.sqrt(np.mean(x ** 2, axis=axis))


def patient_variability(records):
    """Within-patient vs between-patient differences of normal beats (RMS, mV)."""
    templates, within = [], []
    for r in records:
        d = data.extract_beats(r)
        n = d["beats"][d["labels"] == 0]
        if len(n) < 50:
            continue
        t = np.median(n, axis=0)
        templates.append(t)
        within.append(float(np.median(rms(n - t))))
    templates = np.asarray(templates)
    between = [float(rms(templates[i] - templates[j]))
               for i in range(len(templates)) for j in range(i + 1, len(templates))]
    return np.asarray(within), np.asarray(between), templates


def main():
    ds1, ds2 = data.load_split(data.DS1), data.load_split(data.DS2)
    t = np.arange(-data.PRE, data.POST) / data.FS * 1000
    summary = {"DS1": class_counts(ds1), "DS2": class_counts(ds2),
               "records": {"DS1": len(data.DS1), "DS2": len(data.DS2)}}

    # 1. average beat per class (class counts are in the summary / report table)
    fig, axs = plt.subplots(1, 4, figsize=(6.7, 1.9), sharey=True)
    for ax, (i, c) in zip(axs, enumerate(data.CLASSES[:4])):
        b = ds1["beats"][ds1["labels"] == i]
        m, s = b.mean(axis=0), b.std(axis=0)
        ax.fill_between(t, m - s, m + s, color=CLASS_COLORS[c], alpha=0.2, lw=0)
        ax.plot(t, m, color=CLASS_COLORS[c], lw=1.2)
        ax.set_title(f"{c} ({len(b):,} beats)")
        ax.set_xlabel("ms from R peak")
    axs[0].set_ylabel("mV")
    save_fig(fig, "s4_eda_beats")

    # 2. RR timing: premature beats arrive early relative to the local rhythm
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    ratio = ds1["pre_rr"] / ds1["local_rr"]
    for i, c in enumerate(data.CLASSES[:4]):
        ax.hist(ratio[ds1["labels"] == i], bins=np.linspace(0.3, 1.8, 61), density=True,
                histtype="step", lw=1.2, color=CLASS_COLORS[c], label=c)
    ax.set_xlabel("previous RR / local average RR")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    save_fig(fig, "s4_eda_rr")
    summary["median_rr_ratio"] = {c: float(np.median(ratio[ds1["labels"] == i]))
                                  for i, c in enumerate(data.CLASSES[:4])}

    # 3. every patient is different: normal beats of one patient vs other patients
    within, between, templates = patient_variability(data.DS1 + data.DS2)
    fig, axs = plt.subplots(1, 2, figsize=(6.7, 2.3))
    for tp in templates:
        axs[0].plot(t, tp, lw=0.6, alpha=0.6)
    axs[0].set_title("median normal beat of each patient")
    axs[0].set_xlabel("ms from R peak")
    axs[0].set_ylabel("mV")
    bins = np.linspace(0, max(within.max(), between.max()), 40)
    axs[1].hist(within, bins=bins, density=True, alpha=0.6, label="within a patient", color="#1f4e79")
    axs[1].hist(between, bins=bins, density=True, alpha=0.6, label="between patients", color="#c77c02")
    axs[1].set_xlabel("RMS difference of normal beats (mV)")
    axs[1].set_title("normal beats differ more between patients")
    axs[1].legend(frameon=False)
    save_fig(fig, "s4_eda_patients")
    summary["variability_mV"] = {"within_median": float(np.median(within)),
                                 "between_median": float(np.median(between)),
                                 "ratio": float(np.median(between) / np.median(within)),
                                 "patients": int(len(templates))}

    # 4. the three recorded noise types added to a clean segment at 6 dB
    clean = data.bandpass(data.load_signal(100))[30 * data.FS:33 * data.FS]
    ts = np.arange(len(clean)) / data.FS
    fig, axs = plt.subplots(3, 1, figsize=(6.7, 4.0), sharex=True, gridspec_kw={"hspace": 0.55})
    names = {"bw": "baseline wander", "ma": "muscle artefact", "em": "electrode motion"}
    for ax, (k, label) in zip(axs, names.items()):
        n = data.load_noise(k)[:len(clean)]
        n = n - n.mean()
        n *= np.sqrt(np.mean(clean ** 2) / np.mean(n ** 2) / 10 ** (6 / 10))
        ax.plot(ts, clean + n, color="#b85450", lw=0.6, label="noisy (6 dB)")
        ax.plot(ts, clean, color="#1f4e79", lw=0.8, label="clean")
        ax.set_title(label, loc="left")
        ax.set_ylabel("mV")
    axs[0].legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    axs[-1].set_xlabel("time (s)")
    save_fig(fig, "s4_eda_noise")

    save_json(summary, "eda")
    print(summary)


if __name__ == "__main__":
    main()

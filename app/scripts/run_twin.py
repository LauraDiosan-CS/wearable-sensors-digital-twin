"""Calibrate a twin for every record and measure how well its deviation score flags ectopic beats.

usage: python scripts/run_twin.py   (run from the app/ folder)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from cardiactwin import data, twin as tw  # noqa: E402
from cardiactwin.style import CLASS_COLORS, OUT_DIR, plt, save_fig, save_json  # noqa: E402


def auroc(scores, labels, positive):
    """AUROC of a score for separating one class (positive) from normal beats."""
    m = (labels == 0) | (labels == positive)
    y = labels[m] == positive
    return float(roc_auc_score(y, scores[m])) if y.any() and (~y).any() else float("nan")


def main():
    rows, per_beat = {}, {}
    for i, r in enumerate(data.DS1 + data.DS2, 1):
        d = data.extract_beats(r)
        cal = tw.calibrate_record(d)
        rows[r] = {"rmse_twin": float(tw.rms(cal["twin"] - cal["template"])),
                   "rmse_default": float(tw.rms(cal["default"] - cal["template"])),
                   "rr": cal["rr"], "params": cal["params"].tolist()}
        per_beat[r] = {"d_twin": cal["d_twin"], "d_template": cal["d_template"]}
        print(f"[{i}] record {r}: twin RMSE {rows[r]['rmse_twin']:.3f} mV "
              f"(population twin {rows[r]['rmse_default']:.3f} mV)", flush=True)
        if r == 119:
            example = (d, cal)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "twin_deviation.npz",
                        **{f"{r}_{k}": v for r, dd in per_beat.items() for k, v in dd.items()})

    # evaluation on the test patients, after the calibration window
    labels, d_twin, d_tmpl = [], [], []
    for r in data.DS2:
        d = data.extract_beats(r)
        m = d["time_s"] >= data.CALIB_SECONDS
        labels.append(d["labels"][m])
        d_twin.append(per_beat[r]["d_twin"][m])
        d_tmpl.append(per_beat[r]["d_template"][m])
    labels, d_twin, d_tmpl = map(np.concatenate, (labels, d_twin, d_tmpl))
    summary = {
        "fit_rmse_mV": {"twin_median": float(np.median([v["rmse_twin"] for v in rows.values()])),
                        "population_median": float(np.median([v["rmse_default"] for v in rows.values()]))},
        "auroc": {f"{c}_vs_N": {"twin": auroc(d_twin, labels, i), "template": auroc(d_tmpl, labels, i)}
                  for i, c in [(1, "S"), (2, "V")]},
        "median_deviation": {c: float(np.median(d_twin[labels == i])) for i, c in enumerate(data.CLASSES[:4])
                             if (labels == i).any()},
        "records": rows,
    }
    save_json(summary, "twin")
    print({k: v for k, v in summary.items() if k != "records"})

    # figure: one patient's twin and the deviation score per class
    d, cal = example
    t = np.arange(-data.PRE, data.POST) / data.FS * 1000
    fig, axs = plt.subplots(1, 2, figsize=(6.7, 2.5), gridspec_kw={"wspace": 0.35})
    axs[0].plot(t, cal["template"], color="#1f4e79", lw=1.4, label="patient's median beat")
    axs[0].plot(t, cal["default"], color="#7f7f7f", lw=1.0, ls="--", label="population twin")
    axs[0].plot(t, cal["twin"], color="#2e7d32", lw=1.1, label="calibrated twin")
    axs[0].set_xlabel("ms from R peak")
    axs[0].set_ylabel("mV")
    axs[0].set_title("record 119: twin calibration")
    axs[0].legend(frameon=False, ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    bins = np.logspace(-0.3, 1.8, 50)
    for i, c in enumerate(data.CLASSES[:3]):
        v = d_twin[labels == i]
        if len(v):
            axs[1].hist(v, bins=bins, density=True, histtype="step", lw=1.2, color=CLASS_COLORS[c], label=c)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("deviation score $d_k$ (test patients)")
    axs[1].set_ylabel("density")
    axs[1].set_title("deviation from the patient's own twin")
    axs[1].legend(frameon=False)
    save_fig(fig, "s4_twin")


if __name__ == "__main__":
    main()

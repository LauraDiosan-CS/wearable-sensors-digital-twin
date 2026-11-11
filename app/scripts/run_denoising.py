"""Train the denoising autoencoder on DS1 and compare it with a band-pass filter on DS2.

usage: python scripts/run_denoising.py   (run from the app/ folder)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cardiactwin import data, denoise as dn  # noqa: E402
from cardiactwin.style import OUT_DIR, plt, save_fig, save_json  # noqa: E402

TEST_SNRS = (0, 6, 12)
TEST_SEGMENTS = 1500


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = dn.train(dn.DenoisingAutoencoder(), dn.SegmentSampler(data.DS1, rng),
                     dn.NoiseSampler("train", rng), rng, log=lambda m: print(m, flush=True))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "denoiser.pt")

    test_rng = np.random.default_rng(1)
    clean_s, noise_s = dn.SegmentSampler(data.DS2, test_rng), dn.NoiseSampler("test", test_rng)
    results, example = {}, None
    for kind in dn.NOISES:
        for snr in TEST_SNRS:
            clean, noisy, bp = [], [], []
            for _ in range(TEST_SEGMENTS):
                c = clean_s.sample()
                y = dn.make_noisy(c, noise_s.sample(kind), snr, test_rng)
                clean.append(c[dn.PAD:dn.PAD + dn.SEG])
                noisy.append(y[dn.PAD:dn.PAD + dn.SEG])
                bp.append(dn.bandpass_baseline(y))
            clean, noisy, bp = map(np.asarray, (clean, noisy, bp))
            dae = dn.denoise(model, noisy)
            snr_in = dn.snr_db(clean, noisy)
            results[f"{kind}_{snr}"] = {
                "snr_in": float(np.mean(snr_in)),
                "bandpass_gain_db": float(np.mean(dn.snr_db(clean, bp) - snr_in)),
                "dae_gain_db": float(np.mean(dn.snr_db(clean, dae) - snr_in)),
                "bandpass_rmse_mV": float(np.mean(np.sqrt(np.mean((bp - clean) ** 2, axis=1)))),
                "dae_rmse_mV": float(np.mean(np.sqrt(np.mean((dae - clean) ** 2, axis=1)))),
            }
            print(kind, snr, results[f"{kind}_{snr}"], flush=True)
            if kind == "em" and snr == 6:
                example = (clean[3], noisy[3], bp[3], dae[3])

    save_json(results, "denoising")
    c, y, bp, dae = example
    ts = np.arange(dn.SEG) / data.FS
    fig, axs = plt.subplots(2, 1, figsize=(6.7, 3.6), sharex=True, sharey=True, gridspec_kw={"hspace": 0.35})
    axs[0].plot(ts, y, color="#b85450", lw=0.7, label="noisy (electrode motion, 6 dB)")
    axs[0].plot(ts, c, color="#1f4e79", lw=0.9, label="clean")
    axs[1].plot(ts, bp, color="#7f7f7f", lw=0.8, label="band-pass filter")
    axs[1].plot(ts, dae, color="#2e7d32", lw=0.9, label="denoising autoencoder")
    axs[1].plot(ts, c, color="#1f4e79", lw=0.6, ls="--", label="clean")
    for ax in axs:
        ax.set_ylabel("mV")
        ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0))  # above the trace
    axs[1].set_xlabel("time (s)")
    save_fig(fig, "s4_denoising_example")


if __name__ == "__main__":
    main()

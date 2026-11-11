"""Signal Monitoring agent, first version: band-pass baseline vs a convolutional denoising autoencoder."""
import numpy as np
import torch
from torch import nn

from . import data

SEG = 512  # samples per segment (about 1.4 s)
PAD = data.FS  # 1 s of context on each side so the band-pass filter has no edge effects
NOISES = ("bw", "em", "ma")


class DenoisingAutoencoder(nn.Module):
    """1-D convolutional encoder-decoder: 512 samples -> 64 x 64 features -> 512 samples."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(16, 32, 9, stride=2, padding=4), nn.ReLU(),
            nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 8, stride=2, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 8, stride=2, padding=3), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 8, stride=2, padding=3))

    def forward(self, x):
        return self.dec(self.enc(x))


def snr_db(clean, estimate):
    return 10 * np.log10(np.sum(clean ** 2, axis=-1) / np.sum((clean - estimate) ** 2, axis=-1))


def make_noisy(clean_long, noise_long, snr, rng):
    """Mix a padded clean segment with noise at a given SNR (measured on the central SEG samples)."""
    n = noise_long - noise_long.mean()
    core = slice(PAD, PAD + SEG)
    scale = np.sqrt(np.mean(clean_long[core] ** 2) / (np.mean(n[core] ** 2) * 10 ** (snr / 10)))
    return clean_long + scale * n


class SegmentSampler:
    """Random clean segments (band-passed MLII) of a set of records, with padding."""

    def __init__(self, records, rng):
        self.signals = [data.bandpass(data.load_signal(r)) for r in records]
        self.rng = rng

    def sample(self):
        s = self.signals[self.rng.integers(len(self.signals))]
        start = self.rng.integers(0, len(s) - SEG - 2 * PAD)
        return s[start:start + SEG + 2 * PAD]


class NoiseSampler:
    """Random noise segments from one half of each noise record (first half: training, second: test)."""

    def __init__(self, half, rng):
        self.noise = {}
        for k in NOISES:
            x = data.load_noise(k)
            mid = len(x) // 2
            self.noise[k] = x[:mid] if half == "train" else x[mid:]
        self.rng = rng

    def sample(self, kind):
        x = self.noise[kind]
        start = self.rng.integers(0, len(x) - SEG - 2 * PAD)
        return x[start:start + SEG + 2 * PAD]


def bandpass_baseline(noisy_long):
    return data.bandpass(noisy_long)[PAD:PAD + SEG]


def to_tensor(segments):
    x = np.asarray(segments, dtype=np.float32)
    scale = x.std(axis=1, keepdims=True) + 1e-6
    return torch.from_numpy((x / scale)[:, None, :]), scale


def train(model, clean_sampler, noise_sampler, rng, epochs=12, per_epoch=12000, batch=64, lr=1e-3, log=print):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        noisy, clean = [], []
        for _ in range(per_epoch):
            c = clean_sampler.sample()
            kind = NOISES[rng.integers(len(NOISES))]
            y = make_noisy(c, noise_sampler.sample(kind), rng.uniform(-6, 18), rng)
            noisy.append(y[PAD:PAD + SEG])
            clean.append(c[PAD:PAD + SEG])
        xn, scale = to_tensor(noisy)
        xc = torch.from_numpy((np.asarray(clean, dtype=np.float32) / scale)[:, None, :])
        perm = torch.randperm(len(xn))
        total = 0.0
        for i in range(0, len(xn), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xn[idx]), xc[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        log(f"epoch {ep + 1}/{epochs}: loss {total / len(xn):.4f}")
    return model


def denoise(model, noisy_segments):
    model.eval()
    x, scale = to_tensor(noisy_segments)
    with torch.no_grad():
        return model(x)[:, 0, :].numpy() * scale

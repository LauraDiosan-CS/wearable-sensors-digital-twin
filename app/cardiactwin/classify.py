"""Arrhythmia Detection agent, first version: feature baseline and a small 1-D CNN (AAMI classes)."""
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn

from . import data

EVAL_CLASSES = ["N", "S", "V", "F"]  # Q has almost no beats in the non-paced records


def morphology(beats, step=6):
    """Downsampled beat shape (42 samples) with the median removed."""
    b = beats[:, ::step]
    return b - np.median(b, axis=1, keepdims=True)


def features(split, extra=None):
    cols = [data.rr_features(split), morphology(split["beats"])]
    if extra is not None:
        cols.append(np.asarray(extra, dtype=np.float32).reshape(len(split["labels"]), -1))
    return np.concatenate(cols, axis=1)


def train_baseline(x, y, seed=0):
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, class_weight="balanced",
                                           early_stopping=False, random_state=seed)
    return model.fit(x, y)


class BeatCNN(nn.Module):
    """Three convolution blocks on the beat shape, then joined with the RR features."""

    def __init__(self, n_rr=5, n_classes=len(data.CLASSES)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(64 + n_rr, 64), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(64, n_classes))

    def forward(self, beat, rr):
        return self.head(torch.cat([self.conv(beat).squeeze(-1), rr], dim=1))


def _cnn_inputs(split, rr_mean, rr_std):
    b = split["beats"] - np.median(split["beats"], axis=1, keepdims=True)
    rr = (data.rr_features(split) - rr_mean) / rr_std
    return torch.from_numpy(b[:, None, :].astype(np.float32)), torch.from_numpy(rr.astype(np.float32))


def train_cnn(train_split, seed=0, epochs=8, batch=128, lr=1e-3, log=print):
    torch.manual_seed(seed)
    rr = data.rr_features(train_split)
    rr_mean, rr_std = rr.mean(axis=0), rr.std(axis=0) + 1e-6
    xb, xr = _cnn_inputs(train_split, rr_mean, rr_std)
    y = torch.from_numpy(train_split["labels"].astype(np.int64))
    counts = np.bincount(train_split["labels"], minlength=len(data.CLASSES)).astype(np.float32)
    weights = torch.from_numpy(np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
                               .astype(np.float32))
    model = BeatCNN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y))
        total = 0.0
        for i in range(0, len(y), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xb[idx], xr[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        log(f"seed {seed} epoch {ep + 1}/{epochs}: loss {total / len(y):.4f}")
    model.rr_norm = (rr_mean, rr_std)
    return model


def predict_cnn(model, split):
    model.eval()
    xb, xr = _cnn_inputs(split, *model.rr_norm)
    with torch.no_grad():
        return model(xb, xr).argmax(dim=1).numpy()


def metrics(y_true, y_pred):
    """Per-class sensitivity (Se), positive predictivity (+P) and F1, plus macro-F1 over N, S, V, F."""
    out = {}
    for c in EVAL_CLASSES:
        i = data.CLASSES.index(c)
        tp = int(np.sum((y_pred == i) & (y_true == i)))
        fn = int(np.sum((y_pred != i) & (y_true == i)))
        fp = int(np.sum((y_pred == i) & (y_true != i)))
        se = tp / (tp + fn) if tp + fn else 0.0
        pp = tp / (tp + fp) if tp + fp else 0.0
        out[c] = {"Se": se, "+P": pp, "F1": 2 * se * pp / (se + pp) if se + pp else 0.0, "n": tp + fn}
    out["macro_F1"] = float(np.mean([out[c]["F1"] for c in EVAL_CLASSES]))
    out["accuracy"] = float(np.mean(y_true == y_pred))
    return out


def after_calibration(split):
    """Mask of beats after the first CALIB_SECONDS of each record (the twin uses those to calibrate)."""
    return split["time_s"] >= data.CALIB_SECONDS


def subset(split, mask):
    return {k: v[mask] for k, v in split.items()}

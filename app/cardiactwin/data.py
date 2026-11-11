"""MIT-BIH Arrhythmia and Noise Stress Test data (PhysioNet): download, beats, splits."""
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

APP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = APP_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
FS = 360  # Hz, both databases

# Inter-patient split of de Chazal et al. (2004); paced records 102, 104, 107, 217 are excluded.
# Record 202 is dropped from DS2 because it comes from the same subject as record 201 (DS1).
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# MIT-BIH beat annotation -> AAMI EC57 class
AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V",
        "F": "F",
        "/": "Q", "f": "Q", "Q": "Q"}
CLASSES = ["N", "S", "V", "F", "Q"]

PRE, POST = int(0.25 * FS), int(0.45 * FS)  # beat window around the R peak (90 + 162 samples)
CALIB_SECONDS = 300  # first 5 minutes of a record are used to calibrate the twin
LOCAL_RR_BEATS = 10


def download():
    """Download both databases from PhysioNet (about 120 MB) into app/data/."""
    for db, probe in (("mitdb", "100.dat"), ("nstdb", "em.dat")):
        out = DATA_DIR / db
        if (out / probe).exists():
            print(f"{db}: already downloaded")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"{db}: downloading ...")
        wfdb.dl_database(db, str(out))


def bandpass(x, lo=0.5, hi=40.0, fs=FS, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def load_signal(record, db="mitdb", lead="MLII"):
    rec = wfdb.rdrecord(str(DATA_DIR / db / str(record)))
    ch = rec.sig_name.index(lead) if lead in rec.sig_name else 0
    return rec.p_signal[:, ch].astype(np.float64)


def load_noise(name):
    """One channel of a Noise Stress Test noise record: 'bw', 'em' or 'ma'."""
    return wfdb.rdrecord(str(DATA_DIR / "nstdb" / name)).p_signal[:, 0].astype(np.float64)


def extract_beats(record):
    """Beats (band-passed MLII, PRE+POST samples), AAMI labels and RR features of one record (cached)."""
    cache = CACHE_DIR / f"{record}.npz"
    if cache.exists():
        return dict(np.load(cache))
    sig = bandpass(load_signal(record))
    ann = wfdb.rdann(str(DATA_DIR / "mitdb" / str(record)), "atr")
    keep = [i for i, s in enumerate(ann.symbol) if s in AAMI]
    r = np.asarray(ann.sample)[keep]
    sym = [ann.symbol[i] for i in keep]
    rr = np.diff(r) / FS  # rr[i] is the interval between beat i and beat i+1
    beats, labels, pre_rr, post_rr, local_rr, rpos = [], [], [], [], [], []
    for i in range(1, len(r) - 1):
        if r[i] - PRE < 0 or r[i] + POST > len(sig):
            continue
        prev = rr[max(0, i - LOCAL_RR_BEATS):i]
        beats.append(sig[r[i] - PRE:r[i] + POST])
        labels.append(CLASSES.index(AAMI[sym[i]]))
        pre_rr.append(rr[i - 1])
        post_rr.append(rr[i])
        local_rr.append(prev.mean())
        rpos.append(r[i])
    out = {
        "beats": np.asarray(beats, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int8),
        "pre_rr": np.asarray(pre_rr, dtype=np.float32),
        "post_rr": np.asarray(post_rr, dtype=np.float32),
        "local_rr": np.asarray(local_rr, dtype=np.float32),
        "time_s": np.asarray(rpos, dtype=np.float32) / FS,
        "rr_all": rr.astype(np.float32),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def load_split(records):
    """Concatenate the beats of several records; adds a 'record' array."""
    parts = {r: extract_beats(r) for r in records}
    keys = ["beats", "labels", "pre_rr", "post_rr", "local_rr", "time_s"]
    out = {k: np.concatenate([parts[r][k] for r in records]) for k in keys}
    out["record"] = np.concatenate([np.full(len(parts[r]["labels"]), r) for r in records])
    return out


def rr_features(d):
    """RR-interval features: pre, post, local average and the two ratios to the local average."""
    pre, post, loc = d["pre_rr"], d["post_rr"], d["local_rr"]
    return np.stack([pre, post, loc, pre / loc, post / loc], axis=1).astype(np.float32)

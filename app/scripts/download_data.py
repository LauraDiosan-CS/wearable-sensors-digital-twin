"""Download the MIT-BIH Arrhythmia and Noise Stress Test databases and cache the beats.

usage: python scripts/download_data.py   (run from the app/ folder)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cardiactwin import data  # noqa: E402


def main():
    data.download()
    records = data.DS1 + data.DS2
    for i, r in enumerate(records, 1):
        d = data.extract_beats(r)
        print(f"[{i}/{len(records)}] record {r}: {len(d['labels'])} beats", flush=True)


if __name__ == "__main__":
    main()

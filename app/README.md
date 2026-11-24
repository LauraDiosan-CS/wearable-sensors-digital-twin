# CardiacTwin app

Code for the AI-enabled cardiac digital twin (Advanced Machine Learning, 2026-2027).

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py
```
or
```bash
uv run scripts/download_data.py
```

`download_data.py` downloads the MIT-BIH Arrhythmia and MIT-BIH Noise Stress Test databases from
PhysioNet (about 120 MB) into `data/` and caches the extracted beats. `data/` and `outputs/` are not
committed.

## Seminar 4 experiments

Run from this folder, in this order:

| Script | Agent | What it does |
|---|---|---|
| `scripts/eda.py` | data | class balance, beat shapes, RR timing, patient variability, noise examples |
| `scripts/run_denoising.py` | Signal Monitoring | band-pass filter vs convolutional denoising autoencoder |
| `scripts/run_classification.py` | Arrhythmia Detection | feature baseline vs 1-D CNN, inter-patient split |
| `scripts/run_forecasting.py` | Physiological Prediction | next-RR forecasting: persistence vs autoregression |
| `scripts/run_twin.py` | Twin Calibration | per-patient twin fit and deviation score |
| `scripts/run_ablation.py` | Twin + Detection | classifier with and without the deviation score |

Figures go to `../documentation/Fig/`, numbers to `outputs/*.json`.

## Multi-agent system

`cardiactwin/agents/` holds the agents and the orchestrator; the shared contract is `agents/state.py`.
Run the system with `python scripts/run_agents.py [record ...]` (default: all DS2 records); reports go
to `outputs/agents/`.

## Tests

`tests/test_twin_fit.py` checks the twin calibration on 9 synthetic beats with known parameters
(3 heart rates); `tests/test_agents_contract.py` checks the agent contract on a hand-made window.
Neither needs downloaded data. Run `pytest tests`.

## Layout

```
cardiactwin/
  data.py       loading, AAMI classes, inter-patient split (DS1 train / DS2 test), beats, RR features
  style.py      plotting style and output folders
  denoise.py    Signal Monitoring agent
  classify.py   Arrhythmia Detection agent
  forecast.py   Physiological Prediction agent
  twin.py       Twin Calibration agent (McSharry et al. 2003 ECG model)
  agents/       shared state, orchestrator, one file per agent
scripts/        one script per experiment
tests/          checks that need no downloaded data
```

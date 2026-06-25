# Microstructure-Grounded Uncertainty for Limit Order Book Forecasting

Code accompanying the paper **"Microstructure-Grounded Uncertainty for Limit
Order Book Forecasting: Regime-Adaptive Calibration and Conformal
Adverse-Selection Filtering"** (manuscript under review, *IEEE Transactions on
Neural Networks and Learning Systems*).

> This repository contains the **source code and experiment drivers only**. The
> manuscript, figures, literature PDFs, trained checkpoints, and the FI-2010 data
> are intentionally **not** included — the datasets are redistributed by their
> original providers (see [Data](#data)).

## Overview

We study uncertainty quantification (UQ) for deep sequence models applied to
high-frequency limit order book (LOB) mid-price movement forecasting (3-class:
up / stationary / down). The contributions are methodological:

- **Leakage-free FI-2010 benchmark** — a widely used concatenated training
  protocol duplicates the cumulative anchored-fold training windows ~4.66× and
  inflates accuracy. We re-benchmark six architectures under a strict anchored
  walk-forward (best accuracy ~68%, vs the ~85–88% commonly reported).
- **EvLOB** (primary) — a single-pass, microstructure-gated *evidential*
  architecture that decomposes epistemic (Dirichlet vacuity) and aleatoric
  uncertainty in one forward pass, with no Monte-Carlo sampling or ensembles. A
  *corrected* order-flow gate improves both accuracy and calibration over ungated
  and naively-gated baselines.
- **Learned conformal gate (cautionary result)** — a compact network mapping
  order-flow microstructure to a per-instance nonconformity adjustment on frozen
  logits. Under the leaky concatenated protocol it appeared to cut the per-regime
  (Mondrian) coverage gap by ~90%; under the leakage-free protocol the base
  classifiers' gaps are already small (0.004–0.014 vs 0.045), so the benefit
  largely vanishes — the gap was an artifact of leakage.
- **Multi-method UQ benchmark** — temperature scaling, MC dropout, and split
  conformal across the benchmarked LOB architectures, including adaptive
  conformal inference (ACI / AgACI) baselines that restore coverage under
  temporal drift, plus LAC / APS / RAPS nonconformity scores and a Mondrian
  per-regime analysis.
- **Honest negative result** — a trajectory-based H¹ path-energy UQ proxy (from
  the Neural CDE / Neural SDE hidden dynamics) is reported transparently as
  below-chance, delimiting where trajectory UQ does and does not help.

Primary benchmark: **FI-2010** (NoAuction / Z-score normalisation). We
additionally report zero-shot cross-market calibration transfer (crypto /
equity) and leave-one-stock-out calibration on FI-2010.

## Repository structure

```
uncertainty-lob/
├── src/                  # Library: models, calibration, data, features, regime, backtest
│   ├── models/           #   DeepLOB, TCN, LOB-Transformer, CNN-LSTM-Attn,
│   │                     #   Neural CDE, EvLOB, LOB-SDE, RAC-MoE
│   ├── calibration/      #   temperature scaling, MC dropout
│   ├── data/             #   FI-2010 + Bybit loaders
│   ├── features/         #   LOB / microstructure feature extraction
│   ├── regime/           #   HMM volatility-regime detector
│   ├── backtest/         #   market-making backtest (adverse-selection penalty)
│   └── train.py
├── experiments/          # Self-contained drivers (training, calibration, conformal, UQ, ablations)
├── compute/              # Cloud-GPU runners + setup (Kaggle / RunPod / Colab)
├── configs/              # training / models / evaluation YAMLs
├── notebooks/            # Exploratory scaffolding (00–08)
├── results/              # Numeric results behind the paper's tables (JSON / CSV) — no large binaries
├── requirements.txt
├── environment.yml
└── .env.example          # Data-path + credential template (copy to .env)
```

## Models

| Model | File | Notes |
|-------|------|-------|
| DeepLOB | `src/models/deeplob.py` | CNN + inception + LSTM |
| TCN | `src/models/tcn.py` | Dilated temporal convolutions |
| LOB-Transformer | `src/models/lob_transformer.py` | Self-attention over the LOB window |
| CNN-LSTM-Attention | `src/models/cnn_lstm_attention.py` | Convolutional front-end + attention |
| Neural CDE | `src/models/neural_cde.py` | Continuous-time controlled dynamics; H¹ trajectory proxy |
| **EvLOB** | `src/models/evlob.py` | Single-pass evidential; epistemic + aleatoric |
| **LOB-SDE** | `src/models/lob_sde.py` | Neural SDE; fixed-step Euler–Maruyama (Δt=0.05, T=1) |
| RAC-MoE | `src/models/rac_moe.py` | Regime-adaptive mixture (exploratory) |

Conformal machinery (split / Mondrian / APS / RAPS + the learned microstructure
gate) lives in `experiments/learned_gate.py`,
`experiments/aps_raps_robustness.py`, `experiments/micro_conformal.py`, and
`experiments/crossmarket_conformal.py`.

## Data

The experiments use the public **FI-2010** benchmark (NoAuction / Z-score; 40
features, window T=100, horizons k ∈ {1, 2, 3, 5, 10}, 3-class labels) and, for
cross-market transfer, **Bybit BTCUSDT** L2 order book data. Neither dataset is
redistributed here. FI-2010 is available on Kaggle (`ulfricirons/fi-2010`); see
`compute/setup.sh` and `src/data/fi2010_loader.py` for the expected layout. Set
paths in `.env` (copy from `.env.example`).

## Installation

```bash
# pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# or conda
conda env create -f environment.yml
conda activate dl_paper

cp .env.example .env   # then edit data paths / credentials
```

## Reproducing the experiments

Each driver in `experiments/` is self-contained. Representative entry points:

```bash
# Train a base classifier (FI-2010, horizon k=10)
python experiments/run_deeplob.py

# Calibration + UQ (temperature scaling, MC dropout)
python experiments/run_calibration.py

# Split conformal + learned microstructure gate
python experiments/learned_gate.py

# APS / RAPS robustness of the gate
python experiments/aps_raps_robustness.py

# Adaptive conformal (ACI / AgACI) under temporal drift
python experiments/aci_recalibration.py

# Cross-market calibration transfer
python experiments/crossmarket_conformal.py

# LOB-SDE 3-seed metrics from saved logits
python experiments/compute_lobsde_3seed.py
```

GPU runs (Neural CDE / SDE, LOB-Transformer v2) were executed on free **Kaggle
T4** runtimes; see `compute/KAGGLE_SETUP.md` and `compute/RUNPOD_SETUP.md`.
Numeric outputs land in `results/` — the JSON/CSV files committed here are the
numbers behind the paper's tables; large tensors and checkpoints are not
committed.

## Citation

```bibtex
@article{dwivedi2026microstructure,
  author  = {Dwivedi, Paritosh and Tiwari, Arti},
  title   = {Microstructure-Grounded Uncertainty for Limit Order Book
             Forecasting: Regime-Adaptive Calibration and Conformal
             Adverse-Selection Filtering},
  journal = {Manuscript under review, IEEE Transactions on Neural Networks
             and Learning Systems},
  year    = {2026}
}
```

## Authors

- **Paritosh Dwivedi** (first author) — VIT University, Vellore, India
- **Arti Tiwari** (corresponding author) — VIT University, Vellore, India ·
  `arti.tiwari@vit.ac.in`

## License

Private repository — all rights reserved. The code will be released under an
open-source license upon publication of the paper.

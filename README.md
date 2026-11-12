# Microstructure-Grounded Uncertainty for Limit Order Book Forecasting

Code accompanying the paper **"Microstructure-Grounded Uncertainty for Limit
Order Book Forecasting: A Single-Pass Evidential Model and a Leakage-Free
Benchmark"** (manuscript under review, *IEEE Transactions on Neural Networks and
Learning Systems*).

> This repository contains the **source code and experiment drivers only**. The
> manuscript, figures, literature PDFs, trained checkpoints, and the FI-2010 data
> are intentionally **not** included; the datasets are redistributed by their
> original providers (see [Data](#data)).

## Overview

We study uncertainty quantification (UQ) for deep sequence models applied to
high-frequency limit order book (LOB) mid-price movement forecasting (3-class:
up / stationary / down). The contributions are methodological:

- **Leakage-free FI-2010 benchmark.** A widely used concatenated training
  protocol duplicates the cumulative anchored-fold training windows 4.66× and
  inflates accuracy. Six architectures are re-benchmarked under a strict
  anchored walk-forward (nine folds, three seeds); the best reaches 68.9%,
  against the 85–88% commonly reported.
- **EvLOB (primary).** A single-pass, microstructure-gated *evidential*
  architecture that reports epistemic (Dirichlet vacuity) and aleatoric
  uncertainty in one forward pass, with no Monte Carlo sampling or ensembles.
  Over 27 paired runs, a *corrected* order-flow gate significantly lowers the
  negative log-likelihood (0.920 → 0.890) and raises error-detection AUROC
  (0.638 → 0.653) over ungated and naively gated baselines; its accuracy and ECE
  gains are consistent but not significant.
- **UQ benchmark under temporal drift.** Temperature scaling (ECE −69% to −82%)
  and split conformal prediction across five classifiers, with adaptive conformal
  inference (ACI / AgACI) restoring the coverage lost to temporal drift, and
  LAC / APS / RAPS nonconformity scores. Zero-shot cross-market transfer is
  near-random: tiny-budget temperature scaling recalibrates only by flattening
  the forecasts, while split conformal keeps 90–97% coverage with near-full sets.
- **Learned conformal gate (cautionary result).** A compact network mapping
  order-flow microstructure to a nonconformity adjustment on frozen logits
  appears to cut per-regime (Mondrian) miscoverage by up to 90% under the leaky
  protocol with HMM regimes. With a median volatility split the same gate widens
  the leaky gaps, and without leakage the gaps shrink from 0.034–0.050 to
  0.001–0.014, where its effect is mixed.
- **Negative results for trajectory UQ.** The Neural CDE H¹ path energy detects
  misclassification weakly (AUROC 0.461 against 0.759 for softmax confidence),
  and the LOB-SDE path energy is below chance (AUROC 0.356). Both come from
  earlier-protocol runs and are reported as diagnostics.

Primary benchmark: **FI-2010** (NoAuction / Z-score normalisation). We
additionally report zero-shot cross-market transfer to crypto (Bybit
BTC / ETH / SOL) and equity (LOBSTER AAPL) order books.

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
├── results/              # Numeric results behind the paper's tables (JSON / CSV), no large binaries
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
cross-market transfer, **Bybit** L2 order books (BTC / ETH / SOL) and **LOBSTER**
AAPL. None of these datasets is redistributed here. FI-2010 is available on
Kaggle: the anchored campaign used `wzy841127938/fi2010`, and some older scripts
point at `ulfricirons/fi-2010`. See `compute/setup.sh` and
`src/data/fi2010_loader.py` for the expected layout. Set paths in `.env` (copy
from `.env.example`).

**Label order.** FI-2010 classes are 0 = up, 1 = stationary, 2 = down (raw labels
1, 2, 3). The crypto and LOBSTER builders use the reverse order (2 = up), so their
labels are mapped with y → 2 − y before FI-2010-trained models are scored on them.

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
Numeric outputs land in `results/`; the JSON/CSV files committed here are the
numbers behind the paper's tables, and large tensors and checkpoints are not
committed. The leakage-free numbers in `results/leakage_free/` come from the
anchored runner in `compute/protocol_v2/`; the ten files recomputed after the
fold-9 retrains of 2026-11-11 come from `compute/protocol_v2/smokefix_20261111/`,
whose README lists the steps.

## Citation

```bibtex
@article{dwivedi2026microstructure,
  author  = {Dwivedi, Paritosh and Tiwari, Arti},
  title   = {Microstructure-Grounded Uncertainty for Limit Order Book
             Forecasting: A Single-Pass Evidential Model and a
             Leakage-Free Benchmark},
  journal = {Manuscript under review, IEEE Transactions on Neural Networks
             and Learning Systems},
  year    = {2026}
}
```

## Authors

- **Paritosh Dwivedi** (first author), VIT University, Vellore, India
- **Arti Tiwari** (corresponding author), VIT University, Vellore, India ·
  `arti.tiwari@vit.ac.in`

## License

Private repository; all rights reserved. The code will be released under an
open-source license upon publication of the paper.

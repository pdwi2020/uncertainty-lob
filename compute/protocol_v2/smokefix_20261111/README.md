# Fold-9 smoke-test retrains (2026-11-11)

Four of the 162 anchored runs (`tcn`, `lob_transformer`, `lob_transformer_v2`, `rac_moe`, all
fold 9 seed 0) had been left as 3-epoch smoke tests, because `kaggle_campaign.py` skips any run
whose summary says "completed". They were retrained for 80 epochs on a Kaggle T4 with the
unchanged `../anchored_runner.py`, and every leakage-free result that depended on them was
recomputed. The scripts here produced the twelve affected files in `results/leakage_free/`.
`write_results.py` regenerates all twelve byte for byte from the intermediate outputs.

Run data is not in the repository. `common.py` reads it from `LF_RUNS` (original campaign
runs) and `LF_RETRAINED_RUNS` (the four retrains); both default to the paths on the project drive.
Each analysis script takes `orig` or `new`: `orig` reproduces the published JSONs from the
original runs (regression check), `new` swaps in the retrains.

| Step | Script | Output |
|------|--------|--------|
| Retrain on Kaggle | `launch.py <kernel> <model> ...` | run dirs (`summary.json`, logits, `best.pt`) |
| Table I, temperature scaling, LAC, selective prediction | `core_analyses.py new` | `core_new.json` |
| Wilcoxon + Holm over the 27 fold-seed runs (baseline pairs, EvLOB gates) | `stats.py new` | `significance_new.json` |
| ACI / AgACI | `aci_lf.py new` | `aci_new.json` |
| Regime-conditional accuracy, regime TS | `regime_lf.py new` | `regime_new.json` |
| Fold-9 microstructure features | `fold9_features.py` | `fold9_val_features.npz` |
| Stage the conformal inputs | `stage_conformal.py {new,leaky} {vr_median,hmm}` | `stage_*/` |
| MACC ladder, learned gate, APS/RAPS | `conformal_suite.py stage_<...>` | `micro_conformal_results.json`, `learned_gate_results.json`, `aps_raps_robustness.json` in the stage dir |
| Cross-market transfer | `crossmarket_lf.py new deeplob tcn lob_transformer cnn_lstm_attention --flip` | `crossmarket_new_flip.json` |
| Cross-market conformal | `crossmarket_conformal_flip.py` | `crossmarket_conformal_new_flip.json` |
| Write `results/leakage_free/` | `write_results.py` | twelve JSONs |

`leaky_gaps.py` and `hmm_variants.py` are diagnostics. The first gives the per-regime coverage
gaps on the leaky protocol; the second shows that the fold-9 HMM regime labels collapse.

The cross-market target sets label 2 = up, the reverse of FI-2010 (0 = up). `--flip` maps them
before scoring; versions of `crossmarket.json` before 2026-11-11 compared the two unmapped.

# EXP1 Second-Market Conformal — NEGATIVE / NON-VIABLE (2026-06-14)

Attempted: port the FI-2010 MACC-LOB conformal ladder (LAC -> closed-form gate ->
learned gate; ACI/AgACI) to crypto BTC/ETH/SOL using saved logits + features.

## Why it does not work on the available crypto data
1. **No microstructure variation.** After per-symbol Z-scoring, the window-end
   spread has std ~1e-4 (near-constant). The 7-dim microstructure vector m(x) is
   therefore near-constant, so:
   - the 2-state HMM collapses (e.g. BTC: 1 vs 49,999) -> no regime gap to reduce;
   - the learned gate's feature standardization divides by ~0 -> sigmoid saturates
     -> empty prediction sets -> 0.000 coverage on eval.
2. **Near-chance base models.** FI-2010-trained models are 34-49% accurate
   zero-shot on crypto (already reported in the paper). Conformal sets on a
   near-chance classifier are uninformative (2-3 classes), and ACI under-adapts
   (0.72-0.85 coverage) over the highly non-stationary 1-Hz stream.

## Decision
Do NOT include a second-market conformal/learned-gate table. The paper's existing
crypto treatment (tiny-budget TEMPERATURE recalibration on probabilities, which
needs no microstructure) remains the honest, defensible cross-market scope.
Single-dataset limitation stays acknowledged (LOBench/LOBFrame cited).

Artifacts retained for transparency: crossmarket_conformal.py,
crossmarket_conformal_results.json, crossmarket_conformal_table.csv.

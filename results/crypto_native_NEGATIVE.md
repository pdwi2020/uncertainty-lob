# Native-Trained Crypto Second-Market Probe — NEGATIVE (2026-06-15)

**Question.** The paper's crypto results are zero-shot (FI-2010 models applied to crypto),
which is near-chance. Would *natively training* on crypto LOB yield a meaningful
second-market model (and thus a genuine second-dataset UQ validation)?

**Setup.** `experiments/crypto_native_probe.py`. Pooled BTCUSDT+ETHUSDT+SOLUSDT
(`crypto_*_k10_features.npz`, FI-2010 40-feature schema, 3-class vol-normalized k=10
labels). 30k windows/symbol; per-symbol temporal 80/20 split (72k train / 18k test, no
leakage); per-channel standardization on train; compact dilated-TCN (3 blocks, dilations
1/2/4, 64 ch); 12 epochs; MPS.

**Result — NO-GO.**

| Metric | Value |
|---|---|
| Pooled test accuracy | 0.5535 |
| Pooled Cohen's κ | **0.035** |
| Pooled macro-F1 | **0.286** |
| Chance (3-class) | acc 0.333, κ 0 |

Per symbol: BTC acc 0.583/κ 0.017 · ETH acc 0.476/κ 0.049 · SOL acc 0.601/κ 0.006.

**Interpretation.** The 55% accuracy is a **class-imbalance artifact** — the stationary
class is 56–66% of windows (BTC 64%, ETH 56%, SOL 66%), so the model collapses to the
majority class (macro-F1 0.286 < balanced-chance 0.333; κ ≈ 0). Native training does **not**
beat the majority-class baseline.

**Conclusion.** The cross-market limitation is **intrinsic** to short-horizon crypto LOB
direction labeling at accessible (free) data scales, not merely a zero-shot transfer
failure. This is consistent with the earlier `crossmarket_conformal_NEGATIVE.md` (degenerate
Z-scored microstructure) and with the paper's existing zero-shot Table. We therefore do
**not** add a crypto second-dataset result; the single-trained-dataset scope stands, now
additionally supported by (a) the FI-2010 leave-one-stock-out coverage-transfer result and
(b) this native-training negative. Premium multi-day LOB data (LOBSTER/TAQ) remains the path
to a true second trained market (out of scope: no premium access).

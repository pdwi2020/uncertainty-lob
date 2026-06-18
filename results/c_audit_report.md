# C-Audit: Calibration/Conformal Split Convention
*Generated: 2026-06-10*

## Summary

| Table | Content | Partition used | n | Status |
|-------|---------|----------------|---|--------|
| Table 1 | Baseline accuracy | TEST (Day 9–10) | 354,725 | ✅ Correct |
| Table 2 | DeepLOB horizon sweep | TEST | 354,725 | ✅ Correct |
| Table 3 | ECE / temperature scaling | **VAL (Day 8)** | 337,974 | ⚠️ Val-eval (T* fit on val, ECE on val) |
| Table 4 | Split-conformal (LAC) | **VAL (Day 8)** | 337,974 | ❌ Transductive — see below |
| Table 5 | Selective prediction | TEST | 354,725 | ✅ Correct |
| Table 6 | Regime-conditional | TEST | 354,725 | ✅ Correct |
| Table 7 | Domain shift FI→Bybit | TEST (Bybit) | varies | ✅ Correct |
| Table 8 | Backtest | TEST | 354,725 | ✅ Correct |
| Bootstrap CIs | Acc/F1/κ/ECE | **VAL** | 337,974 | ⚠️ Labeled as val |

## Critical Finding: Transductive Conformal in Table 4

`run_conformal.py` loads `{model}_k10_logits.pt` (VAL partition, n=337,974) and calls `evaluate(logits, labels, alpha)` which **fits q̂ from the same data it evaluates coverage on**.

Evidence: every single model×horizon reports coverage = **exactly** 1−α (e.g., 0.9000 at α=0.10).
This is trivially satisfied by construction — it is NOT a meaningful empirical finding.

The singleton rates and set sizes are still informative (they reflect how confident the model is), but the coverage result is circular.

## Fix Plan (after C-stat #36 generates test logits)

### Table 3 (calibration)
- Keep T* fitted on VAL logits (correct: calibration parameters are fit on held-out val)
- Evaluate ECE on **TEST** logits with the val-fitted temperature
- Report "T* fitted on val partition; ECE evaluated on test partition"

### Table 4 (conformal)
- Fit q̂ = conformal_threshold on **VAL** logits (calibration split)
- Evaluate coverage/set-size on **TEST** logits (test split)
- This gives the proper inductive split conformal result: P(y ∈ C(x)) ≥ 1−α on unseen test data

### Bootstrap CIs
- After test logits available: recompute CIs on TEST partition
- The val CIs (Table 3) remain valid as a reference for calibration quality on the held-out set

## Impact on Current Draft
Tables 1, 5, 6, 7, 8: **no change needed** (already on test).
Tables 3 and 4: will be updated after C-stat. The *rankings and qualitative findings* are expected to be stable (coverage will remain ≥ 1−α; singleton rates similar). But numbers will change slightly.

## Action Items
- [ ] C-stat (#36): push Kaggle kernel to save test-partition logits for DeepLOB, TCN, LOBTransformer, CNN-LSTM-Attn
- [ ] After C-stat: update run_calibration.py to evaluate ECE on test logits with val-fitted T*
- [ ] After C-stat: update run_conformal.py to separate cal set (val) from eval set (test)
- [ ] Paper note: add footnote to Tables 3/4 clarifying "T* and q̂ fitted on Day 8 (validation); ECE and coverage evaluated on Days 9–10 (test)"

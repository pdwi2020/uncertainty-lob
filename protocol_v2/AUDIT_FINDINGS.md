# Protocol Audit Findings

Generated from the existing `results/data_cache` matrices on 2026-06-18.

## Anchored fold evidence

| Fold | Train samples | Test samples | Train increment | Equals previous test |
|---:|---:|---:|---:|:---:|
| 1 | 39,512 | 38,397 | - | - |
| 2 | 77,909 | 28,535 | 38,397 | yes |
| 3 | 106,444 | 37,023 | 28,535 | yes |
| 4 | 143,467 | 34,785 | 37,023 | yes |
| 5 | 178,252 | 39,152 | 34,785 | yes |
| 6 | 217,404 | 37,346 | 39,152 | yes |
| 7 | 254,750 | 55,478 | 37,346 | yes |
| 8 | 310,228 | 52,172 | 55,478 | yes |
| 9 | 362,400 | 31,937 | 52,172 | yes |

The training files are cumulative anchored folds. Concatenating them creates
1,690,366 rows from a final-fold training set of 362,400 rows, an inflation of
4.66x. Protocol V2 therefore treats every `(Train_CF_i, Test_CF_i)` pair as an
independent experiment and never concatenates cumulative folds.

## Additional validity issues isolated by Protocol V2

- Legacy and corrected microstructure features are separate named variants.
- Corrected features use the documented ask-price, ask-volume, bid-price,
  bid-volume layout and temporal changes for OFI proxies.
- Decimal-precision microstructure inputs are accepted only after exact sample
  count and label-alignment checks against the ZScore model inputs.
- The original `results/` directory is never used as an output destination.

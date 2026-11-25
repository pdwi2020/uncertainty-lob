# Protocol V2

This directory contains an isolated FI-2010 training pipeline. It does not read
or write the legacy `results/` directory.

## Invariants

- One anchored FI-2010 training fold is paired with its matching test fold.
- Cumulative training folds are never concatenated.
- Sliding windows never cross file boundaries.
- Hyperparameters are selected without reading the paired test labels.
- Every run has an immutable configuration, JSONL metrics, status heartbeat,
  resumable `last.pt`, and validation-selected `best.pt` checkpoint.
- The legacy, corrected, and disabled microstructure feature paths are explicit
  experiment variants.

## Entry points

```powershell
python -m protocol_v2.audit --data-root G:\DL_Paper_v2\data\fi-2010
python -m unittest discover -s protocol_v2\tests -v
python -m protocol_v2.campaign --config protocol_v2\config\campaign_smoke.yaml
python -m protocol_v2.monitor.collector --runs-root G:\DL_Paper_v2\runs
python -m streamlit run protocol_v2\monitor\app.py --server.address 127.0.0.1
```

The long campaign is configured in `config/campaign_long.yaml`. Run the smoke
campaign first and inspect its metrics before starting the long queue.

"""Fold-9 validation microstructure features, toxicity and HMM regimes, aligned to the runner's
validation windows (last 10% of Train_Dst_NoAuction_ZScore_CF_9 rows, window 100, label at the
window's last row). Feature and HMM definitions are imported from build_macc_features.py.

Usage: python fold9_features.py   -> writes fold9_val_features.npz next to this file
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments"))
from build_macc_features import build_hmm_features, compute_toxicity, extract_microstructure  # noqa: E402

FI = Path("/Volumes/Crucial X9/data/market_data/equities/kaggle/fi-2010/BenchmarkDatasets/"
          "NoAuction/1.NoAuction_Zscore")
SEQ, VAL_FRAC, HCOL = 100, 0.1, 148          # k=10 label row in the raw (149, N) matrix


def load(path):
    raw = np.loadtxt(path)
    return np.ascontiguousarray(raw[:40].T, dtype=np.float64), raw[HCOL].astype(np.int64) - 1


def main():
    from hmmlearn import hmm
    feat, lab = load(FI / "NoAuction_Zscore_Training" / "Train_Dst_NoAuction_ZScore_CF_9.txt")
    split = int(len(feat) * (1 - VAL_FRAC))
    train, val = feat[:split], feat[split:]
    n_win = len(val) - SEQ + 1

    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    model.fit(build_hmm_features(train))
    high = int(np.argmax(model.means_[:, 0]))
    r_all = (model.predict(build_hmm_features(val)) == high).astype(np.int8)
    m_all = extract_microstructure(val)
    psi_all = compute_toxicity(m_all)

    sl = slice(SEQ - 1, SEQ - 1 + n_win)           # window i ends at val row i + SEQ - 1
    out = Path(__file__).resolve().parent / "fold9_val_features.npz"
    np.savez(out, r_val=r_all[sl], m_val=m_all[sl], psi_val=psi_all[sl], labels=lab[split:][sl],
             split_row=np.array([split]))
    print("windows", n_win, "regime counts", np.bincount(r_all[sl]), "->", out)


if __name__ == "__main__":
    main()

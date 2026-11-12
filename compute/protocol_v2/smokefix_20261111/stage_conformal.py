"""Stage a results directory for the repo's conformal scripts (micro_conformal, learned_gate,
aps_raps_robustness): fold-9 seed-0 validation logits per base model plus fold-9 features.

Usage: python stage_conformal.py {orig|new|leaky} <regime-rule>
  leaky  stages the concatenated-protocol validation logits and features instead (337,974 windows)
  regime-rule: hmm (fold9_features.npz HMM labels) or vr_median (v_ratio above its calibration-half
  median, i.e. the fold-local median-split proxy applied to the conformal calibration split)
"""
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from common import BASE, run_dir

HERE = Path(__file__).resolve().parent
LEAKY = Path(os.environ.get("LF_LEAKY_RESULTS", "/Volumes/Crucial X9/Research Projects/DL_Paper/results"))


def main():
    which, rule = sys.argv[1], sys.argv[2]
    stage = HERE / f"stage_{which}_{rule}"
    stage.mkdir(exist_ok=True)
    if which == "leaky":
        stage_leaky(stage, rule)
        return
    for m in BASE:
        shutil.copy2(run_dir(m, 9, 0, which == "new") / "validation_logits.pt", stage / f"{m}_k10_logits.pt")
    f = np.load(HERE / "fold9_val_features.npz")
    r = f["r_val"].astype(np.int64)
    if rule == "vr_median":
        vr = f["m_val"][:, 2]
        r = (vr > np.median(vr[: len(vr) // 2])).astype(np.int64)
    np.savez(stage / "macc_features.npz", r_val=r, m_val=f["m_val"], psi_val=f["psi_val"],
             entropy_val=np.zeros(len(r), dtype=np.float32), split_idx=f["split_row"], seq_len=np.array([100]))
    print(stage, "regimes", np.bincount(r[len(r) // 2:]))


def stage_leaky(stage, rule):
    for m in BASE:
        shutil.copy2(LEAKY / f"{m}_k10_logits.pt", stage / f"{m}_k10_logits.pt")
    if rule == "hmm":
        shutil.copy2(LEAKY / "macc_features.npz", stage / "macc_features.npz")
        return
    f = np.load(LEAKY / "macc_features.npz")
    vr = f["m_val"][:, 2]
    r = (vr > np.median(vr[: len(vr) // 2])).astype(np.int64)
    np.savez(stage / "macc_features.npz", r_val=r, m_val=f["m_val"], psi_val=f["psi_val"],
             entropy_val=np.zeros(len(r), dtype=np.float32), split_idx=f["split_idx"], seq_len=f["seq_len"])
    print(stage, "regimes", np.bincount(r[len(r) // 2:]))


if __name__ == "__main__":
    main()

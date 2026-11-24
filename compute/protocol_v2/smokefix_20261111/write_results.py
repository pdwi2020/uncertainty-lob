"""Write the recomputed leakage-free results into results/leakage_free/ of this repository.
Schemas follow the originals where figures read them; the originals are in git history.
Run after the analysis scripts listed in README.md have written their outputs next to this file."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score
from common import ece, load_logits

HERE = Path(__file__).resolve().parent
LF = HERE.parents[2] / "results" / "leakage_free"
NOTE = ("Recomputed 2026-11-11 after retraining the fold-9 seed-0 tcn/lob_transformer/lob_transformer_v2/"
        "rac_moe runs, which had been 3-epoch smoke tests; scripts in X9 runs/protocol_v2_long_smokefix_20261111/")
LABEL_MAPPING = ("Target labels (built as 2=up, 0=down) are mapped with y -> 2-y to the FI-2010 order the "
                 "source models were trained on (raw 1=up, 3=down, i.e. class 0=up, 2=down). Earlier versions "
                 "of this file compared the two conventions unmapped, which swapped up and down.")
CHANGED = ["table2_baseline", "conformal_logit_only", "selective_prediction", "deep_ensemble", "regime_conditional",
           "regime_ts", "aci", "learned_gate_mondrian", "aps_raps", "macc_ladder", "crossmarket"]

core = json.load(open(HERE / "core_new.json"))
reg = json.load(open(HERE / "regime_new.json"))
dump = lambda n, obj: (LF / f"{n}.json").write_text(json.dumps(obj, indent=2))

dump("table2_baseline", core["table2_baseline"])
cl = core["conformal_logit_only"]
for v in cl["temp_scaling"].values():
    v.pop("per_seed", None)
dump("conformal_logit_only", {**cl, "note": NOTE})
dump("selective_prediction", core["selective_prediction"])

def ens(models):
    probs, y0 = [], None
    for m in models:
        for s in range(3):
            z, y = load_logits(m, s, "test", True); y0 = y; probs.append(softmax(z, axis=1))
    p = np.mean(probs, axis=0)
    return {"models": models, "seeds": [0, 1, 2], "fold": 9, "n_test": int(len(y0)),
            "accuracy": round(float((p.argmax(1) == y0).mean()), 4), "ece": round(ece(p, y0), 4),
            "kappa": round(float(cohen_kappa_score(y0, p.argmax(1))), 4)}
dump("deep_ensemble", {"ensemble_4model_3seeds": ens(["deeplob", "tcn", "lob_transformer", "cnn_lstm_attention"]),
                       "ensemble_3model_3seeds": ens(["tcn", "lob_transformer", "cnn_lstm_attention"]), "note": NOTE})
dump("regime_conditional", reg["regime_conditional"])
dump("regime_ts", {"regime_ts": reg["regime_ts"], "note": NOTE})
dump("aci", {"task": "ACI_AgACI_leakage_free", "alpha": 0.1, "fold": 9, "horizon_k": 10,
             "methodology": "Val logits -> LAC q-hat -> ACI(eta=0.05) and AgACI (majority vote of 6 ACI experts, "
                            "eta 0.005-0.2) on the test stream; 3 seeds averaged",
             "models": json.load(open(HERE / "aci_new.json")), "note": NOTE})
st = HERE / "stage_new_vr_median"
dump("learned_gate_mondrian", {"regime_rule": "v_ratio above its calibration-half median (fold-9 validation)",
                               "lambda_star": {m: v["lambda_star"] for m, v in json.load(open(st / "micro_conformal_results.json")).items()},
                               "models": json.load(open(st / "learned_gate_results.json")),
                               "leaky_same_rule": json.load(open(HERE / "stage_leaky_vr_median" / "learned_gate_results.json")),
                               "leaky_original_hmm": json.load(open(HERE / "stage_leaky_hmm" / "learned_gate_results.json")),
                               "note": NOTE})
dump("aps_raps", {**json.load(open(st / "aps_raps_robustness.json")), "note": NOTE})
# the full MACC-LOB conformal ladder (LAC, Tox-CF, LAC + Mondrian, ...) behind the supplement's ladder table
dump("macc_ladder", {"regime_rule": "v_ratio above its calibration-half median (fold-9 validation)",
                     "split": "q-hat fitted on the first half of the fold-9 validation split, coverage on the second",
                     "models": json.load(open(st / "micro_conformal_results.json")), "note": NOTE})
# target markets label 2=up; crossmarket_lf.py --flip maps them to the FI-2010 order (0=up)
cm = json.load(open(HERE / "crossmarket_new_flip.json"))
dump("crossmarket", {"protocol": "leakage_free_anchored_fold9_seed0", "n_windows_per_market": 50000,
                     "recalibration_method": "T fit by NLL on first K target samples over grid [0.5,10] step 0.25; "
                                             "T_unbounded_K200 = uncapped LBFGS optimum", **cm, "note": NOTE,
                     "conformal_lac_alpha0.10": {"method": "q-hat from first n_cal target windows, coverage on remainder",
                                                 "results": json.load(open(HERE / "crossmarket_conformal_new_flip.json"))},
                     "label_mapping": LABEL_MAPPING})
print("written:", ", ".join(CHANGED))

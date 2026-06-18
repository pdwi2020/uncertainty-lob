"""Fill Track-A ?? placeholders in main.tex once training is complete.

Usage:
  python3 fill_track_a_placeholders.py \
      --results-dir /path/to/results \
      --tex-in  main.tex \
      --tex-out main.tex          # overwrites in place
"""
import argparse, json, re
from pathlib import Path

def load_summary(results_dir):
    p = Path(results_dir) / "track_a_full_analysis.json"
    if not p.exists():
        raise FileNotFoundError(f"Run track_a_analysis.py first: {p}")
    return json.load(open(p))

def fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and val != val): return "??"
    return f"{val:.{decimals}f}"

def fmt_ci(triple, decimals=2, pct=False):
    """triple = (mean, lo, hi)"""
    mean, lo, hi = triple
    hw = (hi - lo) / 2
    if pct:
        return f"{mean*100:.{decimals}f}$\\pm${hw*100:.{decimals}f}"
    return f"{mean:.{decimals}f}$\\pm${hw:.{decimals}f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--tex-in",  required=True)
    ap.add_argument("--tex-out", required=True)
    args = ap.parse_args()

    data = load_summary(args.results_dir)

    def get(key, field, pct=False, decimals=2):
        r = data.get(key)
        if r is None: return "??"
        v = r.get(field)
        if v is None: return "??"
        if isinstance(v, (list, tuple)) and len(v) == 3:
            return fmt_ci(v, decimals=decimals, pct=pct)
        if pct:
            return f"{v*100:.{decimals}f}"
        return fmt(v, decimals)

    # ── Numbers to fill ──────────────────────────────────────────────────────
    eg_acc   = get("evlob_gate",   "acc",      pct=True)
    eg_kap   = get("evlob_gate",   "kappa",    decimals=3)
    eg_ece   = get("evlob_gate",   "ece",      decimals=4)
    eg_epi   = get("evlob_gate",   "vacuity",  decimals=4)
    eg_ale   = get("evlob_gate",   "aleatoric", decimals=4)

    en_acc   = get("evlob_nogate", "acc",       pct=True)
    en_kap   = get("evlob_nogate", "kappa",     decimals=3)
    en_ece   = get("evlob_nogate", "ece",       decimals=4)
    en_epi   = get("evlob_nogate", "vacuity",   decimals=4)
    en_ale   = get("evlob_nogate", "aleatoric", decimals=4)

    rm_acc   = get("rac_moe_gate", "acc",      pct=True)
    rm_kap   = get("rac_moe_gate", "kappa",    decimals=3)
    rm_ece   = get("rac_moe_gate", "ece",      decimals=4)

    ls_acc   = get("lob_sde_gate", "acc",       pct=True)
    ls_f1    = get("lob_sde_gate", "f1_macro", decimals=4)
    ls_kap   = get("lob_sde_gate", "kappa",    decimals=3)
    ls_ece   = get("lob_sde_gate", "ece",      decimals=4)

    # Accuracy gap: gate vs backbone (87.6%)
    def _mean(key, field):
        r = data.get(key)
        if r is None: return None
        v = r.get(field)
        if isinstance(v, (list, tuple)): return v[0]
        return v

    eg_mean  = _mean("evlob_gate",   "acc")
    en_mean  = _mean("evlob_nogate", "acc")
    gap_gate = f"{(0.876 - eg_mean)*100:.1f}" if eg_mean else "??"
    delta_ge = f"{(en_mean - eg_mean)*100:.2f}" if (eg_mean and en_mean) else "??"

    tex = open(args.tex_in).read()

    # ── Table rows ───────────────────────────────────────────────────────────
    tex = tex.replace(
        "EvLOB (gate on)  & ??  & ??  & ??  & ??  & ??  \\\\",
        f"EvLOB (gate on)  & {eg_acc} & {eg_kap} & {eg_ece} & {eg_epi} & {eg_ale}  \\\\"
    )
    tex = tex.replace(
        "EvLOB (gate off\\textdagger) & ?? & ?? & ?? & ?? & ?? \\\\",
        f"EvLOB (gate off\\textdagger) & {en_acc} & {en_kap} & {en_ece} & {en_epi} & {en_ale} \\\\"
    )
    tex = tex.replace(
        "RAC-MoE  & ?? & ?? & ?? & -- & -- \\\\",
        f"RAC-MoE  & {rm_acc} & {rm_kap} & {rm_ece} & -- & -- \\\\"
    )
    tex = tex.replace(
        "Full (3 seeds, mean$\\pm$CI) & ?? & ?? & ?? & ?? & -- \\\\",
        f"Full (3 seeds, mean$\\pm$CI) & {ls_acc} & {ls_f1} & {ls_kap} & {ls_ece} & -- \\\\"
    )

    # ── Inline text ──────────────────────────────────────────────────────────
    # Abstract ??
    tex = tex.replace(
        "(??\,\\% test accuracy, 3 seeds)",
        f"({eg_acc}\\,\\% test accuracy, 3 seeds)"
    )
    # Introduction contribution bullet ??
    tex = tex.replace(
        "yield ??\\,\\% test accuracy",
        f"yield {eg_acc}\\,\\% test accuracy"
    )
    # Results section inline text
    tex = tex.replace(
        "EvLOB (gate on) achieves ??\\,\\% test accuracy, representing a ??\\,pp reduction",
        f"EvLOB (gate on) achieves {eg_acc}\\,\\% test accuracy, representing a {gap_gate}\\,pp reduction"
    )
    tex = tex.replace(
        "removing the microstructure gate\nmodestly raises raw accuracy by ??\\,pp",
        f"removing the microstructure gate\nmodestly raises raw accuracy by {delta_ge}\\,pp"
    )
    tex = tex.replace(
        "The gate-off ablation (??\\,\\%)",
        f"The gate-off ablation ({en_acc}\\,\\%)"
    )
    tex = tex.replace(
        "RAC-MoE (??\\,\\%)",
        f"RAC-MoE ({rm_acc}\\,\\%)"
    )

    open(args.tex_out, "w").write(tex)
    print(f"Written → {args.tex_out}")
    print(f"  EvLOB gate: acc={eg_acc}  kappa={eg_kap}  ECE={eg_ece}  vac={eg_epi}")
    print(f"  EvLOB nogate: acc={en_acc}  kappa={en_kap}  ECE={en_ece}")
    print(f"  RAC-MoE: acc={rm_acc}  kappa={rm_kap}  ECE={rm_ece}")
    print(f"  LOB-SDE: acc={ls_acc}  f1={ls_f1}  kappa={ls_kap}  ECE={ls_ece}")

if __name__ == "__main__":
    main()

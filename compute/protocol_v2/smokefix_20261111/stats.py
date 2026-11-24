"""Paired two-sided Wilcoxon over the 27 fold-seed runs: the 15 baseline accuracy pairs and the 12
EvLOB gate tests, Holm within each family. Writes significance_{orig,new}.json next to this file."""
import itertools, json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from common import ALL6, RUNS, run_dir


def accs(model, new):
    return np.array([json.load(open(run_dir(model, f, s, new) / "summary.json"))["test"]["accuracy"]
                     for f in range(1, 10) for s in range(3)])


def holm(pvals):
    order = np.argsort(pvals)
    adj, running = np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def baseline_tests(new):
    a = {m: accs(m, new) for m in ALL6}
    pairs = list(itertools.combinations(ALL6, 2))
    raw = np.array([wilcoxon(a[x], a[y], alternative="two-sided").pvalue for x, y in pairs])
    adj = holm(raw)
    return [{"pair": f"{x} vs {y}", "mean_diff_pp": float((a[x] - a[y]).mean() * 100),
             "wins": int((a[x] > a[y]).sum()), "p": float(r), "p_holm": float(h)}
            for (x, y), r, h in zip(pairs, raw, adj)]


def evlob_gate_tests():
    """EvLOB gate ablation: corrected vs none, corrected vs legacy and legacy vs none on accuracy,
    ECE, NLL and error-detection AUROC, Holm within these 12 tests. The gate-integrity runs were
    not affected by the fold-9 smoke tests, so they are always read from the campaign runs."""
    def vals(gate, metric):
        return np.array([json.load(open(RUNS / f"gate_integrity__evlob__f{f}__k10__s{s}__{gate}__bayes_risk__kl1p0"
                                        / "summary.json"))["test"][metric]
                         for f in range(1, 10) for s in range(3)])
    tests = []
    for a, b in (("corrected", "none"), ("corrected", "legacy"), ("legacy", "none")):
        for metric in ("accuracy", "ece", "nll", "error_auroc"):
            x, y = vals(a, metric), vals(b, metric)
            better = x < y if metric in ("ece", "nll") else x > y      # lower ECE/NLL is better
            tests.append({"pair": f"{a} vs {b}", "metric": metric, "mean_diff": float((x - y).mean()),
                          "better": int(better.sum()), "p": float(wilcoxon(x, y, alternative="two-sided").pvalue)})
    for t, h in zip(tests, holm(np.array([t["p"] for t in tests]))):
        t["p_holm"] = float(h)
    return tests


if __name__ == "__main__":
    new = sys.argv[1] == "new"
    baseline, evlob = baseline_tests(new), evlob_gate_tests()
    for row in baseline:
        print(f"{row['pair']:42s} diff={row['mean_diff_pp']:+6.2f}pp wins={row['wins']:2d}/27 "
              f"p={row['p']:.4g} holm={row['p_holm']:.4g}")
    for row in evlob:
        print(f"{row['pair']:20s} {row['metric']:12s} diff={row['mean_diff']:+.4f} better={row['better']:2d}/27 "
              f"p={row['p']:.4g} holm={row['p_holm']:.4g}")
    (Path(__file__).resolve().parent / f"significance_{sys.argv[1]}.json").write_text(json.dumps(
        {"test": "paired two-sided Wilcoxon over the 27 fold-seed runs at k=10, Holm within each family",
         "baselines": baseline, "evlob_gates": evlob}, indent=1))

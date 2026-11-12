"""Paired two-sided Wilcoxon over the 27 fold-seed accuracies, Holm within the 15 baseline pairs."""
import itertools, json, sys
import numpy as np
from scipy.stats import wilcoxon
from common import ALL6, run_dir


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


if __name__ == "__main__":
    for row in baseline_tests(sys.argv[1] == "new"):
        print(f"{row['pair']:42s} diff={row['mean_diff_pp']:+6.2f}pp wins={row['wins']:2d}/27 "
              f"p={row['p']:.4g} holm={row['p_holm']:.4g}")

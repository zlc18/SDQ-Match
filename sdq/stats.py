"""Paired statistics for common-random-number benchmarks.

Every method is evaluated on the same simulated days, so comparisons are
paired by seed: two-sided Wilcoxon signed-rank test, Holm-Bonferroni
correction across the baselines of one comparison family, a paired bootstrap
95% confidence interval for the relative gain, and the paired effect size
d_z = mean(diff) / sd(diff).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def paired_compare(ours: np.ndarray, base: np.ndarray, n_boot: int = 10000, seed: int = 0):
    ours, base = np.asarray(ours, float), np.asarray(base, float)
    d = ours - base
    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(ours, base, alternative="two-sided", zero_method="wilcox").pvalue)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    rel = ours[idx].mean(1) / base[idx].mean(1) - 1.0
    lo, hi = np.percentile(rel, [2.5, 97.5])
    dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else np.inf
    return dict(diff_mean=float(d.mean()), rel_gain=float(ours.mean() / base.mean() - 1.0),
                ci_lo=float(lo), ci_hi=float(hi), p=p, dz=dz,
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * pvals[i]))
        adj[i] = run
    return adj.tolist()


def summary(x: np.ndarray):
    x = np.asarray(x, float)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return dict(mean=float(x.mean()), std=float(x.std(ddof=1)), median=float(med),
                q1=float(q1), q3=float(q3))

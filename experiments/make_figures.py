"""Rebuild every figure and LaTeX table of the paper from results/*/per_seed.csv.

Usage: python experiments/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval_metrics import analyse, load  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

STYLE = {
    "SDQ-Match": dict(color="#c0392b", marker="o", lw=2.4, zorder=5),
    "Whittle-F": dict(color="#1f4e79", marker="s", ls="--"),
    "PSO-F": dict(color="#6c3483", marker="^", ls="--"),
    "Lag-SIPP": dict(color="#117a65", marker="D", ls="--"),
    "PPO": dict(color="#b9770e", marker="*", ls="--"),
    "Whittle": dict(color="#5dade2", marker="s", ls=":"),
    "PSO": dict(color="#bb8fce", marker="^", ls=":"),
    "GA": dict(color="#d4ac0d", marker="v", ls=":"),
    "DQN": dict(color="#a04000", marker="P", ls=":"),
    "Greedy": dict(color="#45b39d", marker="<", ls=":"),
    "JSQ": dict(color="#7f8c8d", marker="x", ls=":"),
    "Random": dict(color="#aab7b8", marker=".", ls=":"),
    "Erlang-C": dict(color="#566573", marker="|", ls=":"),
}
# every figure uses the same comparison set as the tables
METHODS = list(STYLE)


def mean_q(tag, m):
    d = load(tag)
    return np.mean([v["QASS"] for v in d[m].values()]) if m in d else np.nan


def main():
    done = [t for t in os.listdir(R) if os.path.exists(os.path.join(R, t, "per_seed.csv"))]
    summary = {}
    for t in done:
        summary[t] = analyse(t)
    with open(os.path.join(R, "numbers.json"), "w") as f:
        json.dump(summary, f, indent=1)
    meths = METHODS
    sweeps = [
        ("load", [0.12, 0.14, 0.16, 0.18, 0.20], "Base arrival rate $\\lambda_0$"),
        ("tau", [2, 5, 8, 10], "Reposition latency $\\tau_{rep}$"),
        ("pat", [20, 30, 45, 60], "Mean patience $\\bar P$"),
        ("noise", [0.0, 0.1, 0.2, 0.3, 0.5], "Forecast error $\\epsilon$"),
    ]
    default = {"load": 0.16, "tau": 5, "pat": 45, "noise": 0.0}
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.4))
    for ax, (key, xs, lab), letter in zip(axes.ravel(), sweeps, "abcd"):
        tags = ["main" if x == default[key] else f"{key}_{x}" for x in xs]
        if not all(t in done for t in tags):
            continue
        for m in meths:
            ax.plot(xs, [mean_q(t, m) for t in tags], label=m, ms=3.5, **STYLE.get(m, {}))
        ax.set_xlabel(lab)
        ax.set_ylabel("QASS (%)")
        ax.set_title(f"({letter})", loc="left", fontsize=9)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=7, fontsize=6.8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_sweeps.png"), dpi=300)
    plt.close(fig)
    # ablation bar (main scenario)
    if "main" in summary:
        ab = summary["main"]["ablation"]
        names = {"SDQ-core": "SDQ-Match without\nlearned correction",
                 "abl-no-lead": "core: no lead shift",
                 "abl-steady": "core: steady-state\nErlang-A",
                 "abl-surrogate": "core: surrogate\n$\\pi/(c+1)$",
                 "abl-no-forecast": "core: no forecast\n(observed rate)"}
        keys = [k for k in names if k in ab]
        vals = [-ab[k]["diff_mean"] for k in keys]
        fig, ax = plt.subplots(figsize=(5.2, 2.9))
        cols = ["#c0392b" if v < 0 else "#7f8c8d" for v in vals]
        ax.barh(range(len(keys)), vals, color=cols)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([names[k] for k in keys], fontsize=7.5)
        ax.axvline(0, color="k", lw=0.8)
        for i, k in enumerate(keys):
            p = ab[k]["p"]
            ax.text(vals[i] + (0.1 if vals[i] >= 0 else -0.1), i,
                    f"{vals[i]:+.2f} (p={p:.1g})", va="center",
                    ha="left" if vals[i] >= 0 else "right", fontsize=7)
        ax.set_xlabel("QASS change (points)")
        ax.set_xlim(min(vals) - 2.5, max(1.5, max(vals) + 2.5))
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "fig_ablation.png"), dpi=300)
        plt.close(fig)
    # learning curves of the two ensemble members of the learned correction
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for lab, col, fn in (("member 1 (seed 0)", "#c0392b", "sdq_main_log.json"),
                         ("member 2 (seed 1)", "#2471a3", "sdq_main_s1_log.json")):
        p = os.path.join(R, "models", fn)
        if os.path.exists(p):
            log = json.load(open(p))["log"]
            ax.plot([r["iter"] for r in log], [r["val_qass"] for r in log], color=col, label=lab)
    p = os.path.join(R, "models", "sdq_main_log.json")
    if os.path.exists(p):
        ax.axhline(json.load(open(p))["log"][0]["val_qass"], color="#7f8c8d", ls=":", lw=1,
                   label="queueing core")
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Validation QASS (%)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_learning.png"), dpi=300)
    plt.close(fig)
    print("figures in", FIG)


if __name__ == "__main__":
    main()

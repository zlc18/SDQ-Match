"""Three additional result figures built from the per-day results and traces:
  fig_daily_gain.png - per-day paired QASS gain of SDQ-Match over each baseline (calibration)
  fig_dynamics.png   - time-of-day fleet share, queue and abandonments (30-day means)
  fig_forest.png     - gain over the strongest baseline of each tier in every scenario
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from gen_report import LABELS, T1, T2  # noqa: E402

R = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(R, "figures")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
C_T1, C_T2, C_OURS = "#5d6d7e", "#2471a3", "#c0392b"


def per_seed(tag):
    d = {}
    for r in csv.DictReader(open(os.path.join(R, tag, "per_seed.csv"))):
        d.setdefault(r["method"], {})[int(r["seed"])] = float(r["QASS"])
    return d


def fig_daily_gain():
    d = per_seed("main")
    seeds = sorted(d["SDQ-Match"])
    ours = np.array([d["SDQ-Match"][s] for s in seeds])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw={"width_ratios": [1, 1.5]})
    for ax, tier, col, title in ((axes[0], T2, C_T2, "(a) Tier 2"),
                                 (axes[1], T1, C_T1, "(b) Tier 1")):
        ms = [m for m in tier if m in d]
        gains = {m: 100 * (ours / np.array([d[m][s] for s in seeds]) - 1) for m in ms}
        ms = sorted(ms, key=lambda m: np.median(gains[m]))
        data = [gains[m] for m in ms]
        ax.boxplot(data, vert=False, widths=0.55, showfliers=False,
                   medianprops=dict(color=C_OURS, lw=1.6), boxprops=dict(color=col),
                   whiskerprops=dict(color=col), capprops=dict(color=col))
        rng = np.random.default_rng(0)
        for i, g in enumerate(data):
            ax.scatter(g, i + 1 + rng.uniform(-0.15, 0.15, len(g)), s=7, color=col, alpha=0.55,
                       zorder=3)
            wins = int((g > 0).sum())
            ax.text(1.02, (i + 1 - 0.5) / len(data), f"{wins}/30", transform=ax.transAxes,
                    fontsize=7.5, va="center", color="#333")
        ax.set_yticks(range(1, len(ms) + 1))
        ax.set_yticklabels(ms, fontsize=8)
        if tier == T2:
            ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Per-day QASS gain of SDQ-Match (%)")
        ax.set_title(title, loc="left", fontsize=9)
        if tier == T1:      # every per-day gain over tier 1 is positive: log axis
            ax.set_xscale("log")
            ax.set_xticks([2, 5, 10, 20, 50, 100, 200])
            ax.set_xticklabels(["2", "5", "10", "20", "50", "100", "200"])
    axes[1].text(1.02, 1.02, "wins", transform=axes[1].transAxes, fontsize=7.5)
    axes[0].text(1.02, 1.02, "wins", transform=axes[0].transAxes, fontsize=7.5)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(os.path.join(FIG, "fig_daily_gain.png"), dpi=300)
    plt.close(fig)


def smooth(x, w=15):
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid")


def fig_dynamics():
    z = np.load(os.path.join(R, "traces_main.npz"))
    style = {"SDQ-Match": dict(color=C_OURS, lw=2.0), "Whittle-F": dict(color=C_T2, ls="--"),
             "Lag-SIPP": dict(color="#117a65", ls="--"), "Whittle": dict(color="#5dade2", ls=":")}
    T = z["demand_pm"].shape[1]
    h = np.arange(T) / T * 12 + 7          # map the operating day to 07:00-19:00
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9))
    ax = axes[0]
    ax.plot(h, smooth(z["demand_pm"].mean(0)), color="k", lw=1.2, label="demand share")
    for m, st in style.items():
        ax.plot(h, smooth(z[f"{m}|fleet_pm"].mean(0)), label=m, **st)
    ax.set_ylabel("Afternoon-group share")
    ax.set_title("(a) Fleet vs demand share", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    for m, st in style.items():
        axes[1].plot(h, smooth(z[f"{m}|queue"].mean(0)), **st)
        axes[2].plot(h, z[f"{m}|aband"].mean(0), **st)
    axes[1].set_ylabel("Tourists waiting")
    axes[1].set_title("(b) Total queue", loc="left", fontsize=9)
    axes[2].set_ylabel("Cumulative abandonments")
    axes[2].set_title("(c) Abandonments", loc="left", fontsize=9)
    for ax in axes:
        ax.set_xlabel("Time of day (h)")
        ax.set_xticks([7, 10, 13, 16, 19])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_dynamics.png"), dpi=300)
    plt.close(fig)


def fig_forest():
    nums = json.load(open(os.path.join(R, "numbers.json")))
    rows = [(tag, lab) for tag, lab in LABELS if tag in nums]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 6.6), sharey=True)
    for ax, tier, col, title in ((axes[0], T2, C_T2, "(a) Strongest tier-2 baseline"),
                                 (axes[1], T1, C_T1, "(b) Strongest tier-1 baseline")):
        for i, (tag, lab) in enumerate(rows):
            C = nums[tag]["compare"]
            b = min([m for m in C if m in tier], key=lambda m: C[m]["rel_gain"])
            c = C[b]
            y = len(rows) - 1 - i
            sig = c["p_holm"] < 0.05
            ax.plot([100 * c["ci_lo"], 100 * c["ci_hi"]], [y, y], color=col, lw=1.4)
            ax.scatter([100 * c["rel_gain"]], [y], s=28, zorder=3,
                       facecolor=col if sig else "white", edgecolor=col, linewidth=1.2)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("QASS gain of SDQ-Match (%), 95% CI")
        ax.set_title(title, loc="left", fontsize=9)
    labels = [lab.replace("$", "").replace("\\lambda_0", "λ₀").replace("\\tau", "τ")
              .replace("Patience \\bar P", "Mean patience").replace("\\epsilon", "ε")
              for _, lab in rows]
    labels = ["Calibration" if l.startswith("Calibration") else l for l in labels]
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(labels[::-1], fontsize=7.5)
    axes[1].scatter([], [], s=28, facecolor=C_T1, edgecolor=C_T1, label="$p_{Holm}<0.05$")
    axes[1].scatter([], [], s=28, facecolor="white", edgecolor=C_T1, label="$p_{Holm}\\geq0.05$")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_forest.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig_daily_gain()
    fig_dynamics()
    fig_forest()
    print("figures in", FIG)

"""Summaries and paired significance tests for one benchmark directory.

Usage: python experiments/eval_metrics.py --tag main
Writes results/<tag>/summary.json, results/<tag>/table.tex and prints a report.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.stats import paired_compare, holm, summary  # noqa: E402

METRICS = ["QASS", "SDMR", "AQW", "P95", "AB", "CV", "moves", "ms_per_decision"]


def load(tag):
    path = os.path.join(os.path.dirname(__file__), "..", "results", tag, "per_seed.csv")
    data = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            data[r["method"]][int(r["seed"])] = {k: float(r[k]) for k in METRICS}
    return data


def analyse(tag, ours="SDQ-Match", exclude=("SDQ-core",)):
    data = load(tag)
    methods = list(data)
    seeds = sorted(data[ours])
    out = dict(tag=tag, n_seeds=len(seeds), methods={}, compare={})
    for m in methods:
        out["methods"][m] = {k: summary([data[m][s][k] for s in seeds]) for k in METRICS}
    others = [m for m in methods if m != ours and not m.startswith("abl-") and m not in exclude]
    o = np.array([data[ours][s]["QASS"] for s in seeds])
    comps = {m: paired_compare(o, np.array([data[m][s]["QASS"] for s in seeds])) for m in others}
    adj = holm([comps[m]["p"] for m in others])
    for m, pa in zip(others, adj):
        comps[m]["p_holm"] = pa
        w = paired_compare(-np.array([data[ours][s]["AQW"] for s in seeds]),
                           -np.array([data[m][s]["AQW"] for s in seeds]))
        comps[m]["p_wait"] = w["p"]
    out["compare"] = comps
    # ablations vs full method
    abl = [m for m in methods if m.startswith("abl-") or m in exclude]
    out["ablation"] = {}
    for m in abl:
        # the learned correction is measured against the full method; every other
        # ablation removes one ingredient from the queueing core and is measured against it
        ref = o if (m == "SDQ-core" or "SDQ-core" not in data) else             np.array([data["SDQ-core"][s]["QASS"] for s in seeds])
        out["ablation"][m] = paired_compare(ref, np.array([data[m][s]["QASS"] for s in seeds]))
    best = max(others, key=lambda m: out["methods"][m]["QASS"]["mean"])
    out["strongest_baseline"] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    res = analyse(args.tag)
    d = os.path.join(os.path.dirname(__file__), "..", "results", args.tag)
    with open(os.path.join(d, "summary.json"), "w") as f:
        json.dump(res, f, indent=1)
    M = res["methods"]
    order = sorted(M, key=lambda m: -M[m]["QASS"]["mean"])
    print(f"== {args.tag}: {res['n_seeds']} paired seeds")
    print(f"{'method':16s} {'QASS':>14s} {'median[IQR]':>20s} {'SDMR':>7s} {'AQW':>6s} {'P95':>6s} "
          f"{'AB':>6s} {'ms':>7s}")
    for m in order:
        q = M[m]["QASS"]
        print(f"{m:16s} {q['mean']:6.2f}±{q['std']:5.2f} {q['median']:7.2f}[{q['q1']:.1f},{q['q3']:.1f}]"
              f" {M[m]['SDMR']['mean']:7.2f} {M[m]['AQW']['mean']:6.2f} {M[m]['P95']['mean']:6.1f}"
              f" {M[m]['AB']['mean']:6.2f} {M[m]['ms_per_decision']['mean']:7.3f}")
    print("-- SDQ-Match vs baselines (paired Wilcoxon, Holm)")
    for m, c in sorted(res["compare"].items(), key=lambda kv: kv[1]["rel_gain"]):
        print(f"{m:16s} gain {100*c['rel_gain']:+6.2f}% CI[{100*c['ci_lo']:+.1f},{100*c['ci_hi']:+.1f}]"
              f" p={c['p']:.2e} p_holm={c['p_holm']:.2e} dz={c['dz']:.2f} W/T/L={c['wins']}/{c['ties']}/{c['losses']}")
    print("-- ablations (full minus variant)")
    for m, c in res["ablation"].items():
        print(f"{m:16s} diff {c['diff_mean']:+.2f} ({100*c['rel_gain']:+.2f}%) p={c['p']:.2e}")


if __name__ == "__main__":
    main()

"""Unified benchmark: every method on the same simulated days (common random numbers).

Usage:
  python experiments/run_benchmark.py --scenario main --tag main
  python experiments/run_benchmark.py --scenario main --tag tau8 --over '{"tau_rep": 8}'
  python experiments/run_benchmark.py --scenario main --tag noise30 --noise 0.3

Writes results/<tag>/per_seed.csv (one row per method x seed) and config.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.env import ScenicEnv  # noqa: E402
from sdq.registry import build, BASELINES, OURS  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402

ABLATIONS = ["SDQ-core", "abl-no-lead", "abl-steady", "abl-surrogate", "abl-no-forecast"]


def job(a):
    method, scenario, over, seed, noise, hist_profile, model_scenario, adaptive = a
    cfg = get_scenario(scenario, **over)
    hist_cfg = get_scenario(scenario, **{**over, "profile": hist_profile}) if hist_profile else None
    env = ScenicEnv(cfg)
    pol = build(method, cfg, scenario=model_scenario, noise=noise, hist_cfg=hist_cfg,
                adaptive=adaptive)
    obs = env.reset(seed)
    pol.reset(env, seed)
    done, dt, n = False, 0.0, 0
    while not done:
        t0 = time.perf_counter()
        target = pol.act(obs)
        dt += time.perf_counter() - t0
        n += 1
        obs, _, done, _ = env.step(target)
    m = env.metrics()
    m.update(method=method, seed=seed, ms_per_decision=1000 * dt / n)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="main")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--over", default="{}")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--hist_profile", default=None,
                    help="profile the forecaster's history comes from (mismatch test)")
    ap.add_argument("--model_scenario", default="main")
    ap.add_argument("--no_detector", action="store_true",
                    help="disable the forecaster's pattern-mismatch test (ablation)")
    ap.add_argument("--seeds", default="200-229")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--merge", action="store_true",
                    help="replace only the rows of the listed methods in an existing per_seed.csv")
    args = ap.parse_args()
    over = json.loads(args.over)
    lo, hi = map(int, args.seeds.split("-"))
    seeds = list(range(lo, hi + 1))
    if args.methods == "all":
        methods = [OURS] + BASELINES + ABLATIONS
    else:
        methods = args.methods.split(",")
    tag = args.tag or args.scenario
    outdir = os.path.join(os.path.dirname(__file__), "..", "results", tag)
    os.makedirs(outdir, exist_ok=True)
    jobs = [(m, args.scenario, over, sd, args.noise, args.hist_profile, args.model_scenario,
             not args.no_detector) for m in methods for sd in seeds]
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(job, jobs, chunksize=1)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "per_seed.csv")
    keys = ["method", "seed", "QASS", "SDMR", "AQW", "P95", "AB", "CV", "arrivals", "served",
            "abandoned", "moves", "ms_per_decision"]
    if args.merge and os.path.exists(path):
        keep = [r for r in csv.DictReader(open(path)) if r["method"] not in methods]
        rows = keep + [{k: r[k] for k in keys} for r in rows]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    cfg = get_scenario(args.scenario, **over)
    import scipy
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(dict(args=vars(args), env=cfg.to_dict(), seeds=seeds, methods=methods,
                       python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                       platform=platform.platform(), wall_s=time.time() - t0), f, indent=1)
    print(f"wrote {path} ({len(rows)} rows) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

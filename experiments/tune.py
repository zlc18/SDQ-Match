"""Equal-effort hyperparameter selection for every tunable method.

Each method gets a small grid over its key parameter(s), evaluated on the
validation seeds 100-119 (disjoint from the evaluation seeds 200-229); the
configuration with the highest mean QASS is written to results/tuning.json,
which the registry reads.

Usage: python experiments/tune.py [--scenario main]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq import policies as P  # noqa: E402
from sdq.env import ScenicEnv  # noqa: E402
from sdq.forecast import Forecaster  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402

GRIDS = {
    "Erlang-C": (P.StaticErlangC, dict(beta=[0.0, 0.5, 1.0, 2.0]), False),
    "Whittle": (P.WhittlePolicy, dict(h=[10.0, 20.0, 40.0, 80.0], rate=["observed"]), True),
    "Whittle-F": (P.WhittlePolicy, dict(h=[10.0, 20.0, 40.0, 80.0], rate=["forecast"]), True),
    "PSO": (P.PSOPolicy, dict(horizon=[10, 20, 40], rate=["observed"]), True),
    "PSO-F": (P.PSOPolicy, dict(horizon=[10, 20, 40], rate=["forecast"]), True),
    "GA": (P.GAPolicy, dict(horizon=[10, 20, 40], rate=["observed"]), True),
    "SDQ-core": (P.SDQTransient, dict(gamma1=[1.0, 1.2], gamma2=[1.0, 1.2, 1.5],
                                      H2=[10, 20, 40]), True),
}


def job(a):
    name, kw, scenario, seed = a
    cls, _, fc_needed = GRIDS[name]
    cfg = get_scenario(scenario)
    pol = cls(cfg, Forecaster(cfg), **kw) if fc_needed else cls(cfg, **kw)
    env = ScenicEnv(cfg)
    obs = env.reset(seed)
    pol.reset(env, seed)
    done = False
    while not done:
        obs, _, done, _ = env.step(pol.act(obs))
    return env.metrics()["QASS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="main")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    seeds = list(range(100, 120))
    path = os.path.join(os.path.dirname(__file__), "..", "results", "tuning.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    names = list(GRIDS) if args.methods == "all" else args.methods.split(",")
    with Pool(args.workers) as pool:
        for name in names:
            grid = GRIDS[name][1]
            keys = list(grid)
            res = []
            for vals in itertools.product(*[grid[k] for k in keys]):
                kw = dict(zip(keys, vals))
                q = pool.map(job, [(name, kw, args.scenario, sd) for sd in seeds])
                res.append((float(np.mean(q)), kw))
                print(f"{name} {kw} -> {np.mean(q):.2f}", flush=True)
            best = max(res, key=lambda r: r[0])
            out.setdefault(args.scenario, {})[name] = dict(best=best[1], val_qass=best[0],
                                                           grid=[dict(q=r[0], **r[1]) for r in res])
            with open(path, "w") as f:
                json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

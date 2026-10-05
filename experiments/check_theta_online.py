"""Validation-day check of the online abandonment-rate estimate of SDQ-Match
(theta_hat = abandonments / waiting time, shrunk to the nominal rate).
Validation days 100-119 only. Writes results/check_theta_online.json."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.env import ScenicEnv  # noqa: E402
from sdq.registry import build  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402

SCEN = {
    "main": ({}, None), "load_0.18": ({"lam0": 0.18}, None), "load_0.2": ({"lam0": 0.20}, None),
    "pat_20": ({"mean_patience": 20.0}, None), "pat_60": ({"mean_patience": 60.0}, None),
    "dist_uniformpat": ({"patience_dist": "uniform"}, None),
    "dist_lognormal": ({"service_dist": "lognormal"}, None),
    "dist_negbin": ({"arrival": "negbin"}, None),
    "mismatch_weekend": ({"profile": "weekend"}, "weekday"),
}
VARIANTS = {"off": (False, 0.0), "k200": (True, 200.0), "k50": (True, 50.0)}


def job(a):
    tag, var, seed = a
    over, hist = SCEN[tag]
    cfg = get_scenario("main", **over)
    hist_cfg = get_scenario("main", **{**over, "profile": hist}) if hist else None
    pol = build("SDQ-Match", cfg, hist_cfg=hist_cfg)
    pol.core.theta_online, pol.core.theta_prior = VARIANTS[var]
    env = ScenicEnv(cfg)
    obs = env.reset(seed)
    pol.reset(env, seed)
    done = False
    while not done:
        obs, _, done, _ = env.step(pol.act(obs))
    th = pol.core.theta_hat() / cfg.theta
    return env.metrics()["QASS"], th


def main():
    out = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 14) as pool:
        for tag in SCEN:
            row = {}
            for var in VARIANTS:
                r = pool.map(job, [(tag, var, sd) for sd in range(100, 120)])
                row[var] = dict(qass=float(np.mean([x[0] for x in r])),
                                theta_ratio=float(np.mean([x[1] for x in r])))
            out[tag] = row
            print(tag, {v: (round(d["qass"], 2), round(d["theta_ratio"], 2)) for v, d in row.items()},
                  flush=True)
    for var in VARIANTS:
        print("mean", var, round(float(np.mean([out[t][var]["qass"] for t in SCEN])), 3))
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                     "check_theta_online.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

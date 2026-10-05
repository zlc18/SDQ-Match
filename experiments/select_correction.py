"""Model selection for the learned correction on the validation days (100-119) of
every five-spot scenario. Candidates are given as name=file[+file...] (a '+' builds an
ensemble that averages the members' outputs). Writes results/select_correction.json.

Usage: python experiments/select_correction.py core= main=sdq_main.npz ...
"""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.env import ScenicEnv  # noqa: E402
from sdq.forecast import Forecaster  # noqa: E402
from sdq.registry import load_actor, tuned  # noqa: E402
from sdq.rl import SDQMatch  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402

M = os.path.join(os.path.dirname(__file__), "..", "results", "models")
SCEN = {
    "main": ({}, None, 0.0), "load_0.12": ({"lam0": 0.12}, None, 0.0),
    "load_0.14": ({"lam0": 0.14}, None, 0.0), "load_0.18": ({"lam0": 0.18}, None, 0.0),
    "load_0.2": ({"lam0": 0.20}, None, 0.0),
    "pat_20": ({"mean_patience": 20.0}, None, 0.0), "pat_30": ({"mean_patience": 30.0}, None, 0.0),
    "pat_60": ({"mean_patience": 60.0}, None, 0.0), "noise_0.1": ({}, None, 0.1),
    "noise_0.2": ({}, None, 0.2), "noise_0.3": ({}, None, 0.3), "noise_0.5": ({}, None, 0.5),
    "profile_weekend": ({"profile": "weekend"}, None, 0.0),
    "profile_festival": ({"profile": "festival"}, None, 0.0),
    "mismatch_weekend": ({"profile": "weekend"}, "weekday", 0.0),
    "mismatch_festival": ({"profile": "festival"}, "weekday", 0.0),
    "dist_negbin": ({"arrival": "negbin"}, None, 0.0),
    "dist_lognormal": ({"service_dist": "lognormal"}, None, 0.0),
    "dist_uniformpat": ({"patience_dist": "uniform"}, None, 0.0),
}


def actor_for(spec):
    if not spec:
        return None
    files = spec.split("+")
    acts = tuple(load_actor(os.path.join(M, f))[0] for f in files)
    return acts[0] if len(acts) == 1 else acts


def job(a):
    tag, spec, seed = a
    over, hist, noise = SCEN[tag]
    cfg = get_scenario("main", **over)
    hist_cfg = get_scenario("main", **{**over, "profile": hist}) if hist else None
    pol = SDQMatch(cfg, Forecaster(cfg, noise=noise, hist_cfg=hist_cfg), actor_for(spec),
                   core_kw=tuned("SDQ-core"), place="load", train_tau=5)
    env = ScenicEnv(cfg)
    obs = env.reset(seed)
    pol.reset(env, seed)
    done = False
    while not done:
        obs, _, done, _ = env.step(pol.act(obs))
    return env.metrics()["QASS"]


def main():
    cands = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    res = {}
    with Pool(14) as pool:
        for tag in SCEN:
            res[tag] = {c: float(np.mean(pool.map(job, [(tag, spec, sd) for sd in range(100, 120)])))
                        for c, spec in cands.items()}
            print(tag, {k: round(v, 2) for k, v in res[tag].items()}, flush=True)
    res["_mean"] = {c: float(np.mean([res[t][c] for t in SCEN])) for c in cands}
    res["_candidates"] = cands
    print("mean", {k: round(v, 3) for k, v in res["_mean"].items()})
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                     "select_correction.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

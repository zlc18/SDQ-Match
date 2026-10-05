"""Record time-of-day traces of selected methods at the calibration scenario on all
thirty evaluation days (200-229): fleet committed to the afternoon group, total queue,
cumulative abandonments, and the true afternoon share of demand.
Writes results/traces_main.npz."""
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.demand import spot_groups  # noqa: E402
from sdq.env import ScenicEnv  # noqa: E402
from sdq.registry import build  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402

METHODS = ["SDQ-Match", "Whittle-F", "Lag-SIPP", "Whittle"]
SEEDS = list(range(200, 230))


def job(a):
    method, seed = a
    cfg = get_scenario("main")
    pm = spot_groups(cfg.S, cfg.profile)[1][0]          # afternoon-peak spots
    env = ScenicEnv(cfg)
    pol = build(method, cfg)
    obs = env.reset(seed)
    pol.reset(env, seed)
    T = cfg.T
    fleet_pm = np.zeros(T)
    queue = np.zeros(T)
    aband = np.zeros(T)
    done = False
    while not done:
        target = pol.act(obs)
        t0 = env.t
        obs, _, done, _ = env.step(target)
        for t in range(t0, env.t):
            fleet_pm[t] = obs["committed"][pm].sum() / cfg.N
            queue[t] = env.q_hist[t].sum()
        aband[t0:env.t] = env.n_abandon
    demand_pm = env.day.rates[:, pm].sum(1) / env.day.rates.sum(1)
    return method, seed, fleet_pm, queue, aband, demand_pm


def main():
    jobs = [(m, s) for m in METHODS for s in SEEDS]
    with Pool(12) as pool:
        res = pool.map(job, jobs)
    out = {}
    for m in METHODS:
        rows = [r for r in res if r[0] == m]
        out[f"{m}|fleet_pm"] = np.stack([r[2] for r in rows])
        out[f"{m}|queue"] = np.stack([r[3] for r in rows])
        out[f"{m}|aband"] = np.stack([r[4] for r in rows])
    out["demand_pm"] = np.stack([r[5] for r in res if r[0] == METHODS[0]])
    path = os.path.join(os.path.dirname(__file__), "..", "results", "traces_main.npz")
    np.savez(path, **out)
    print("wrote", path)


if __name__ == "__main__":
    main()

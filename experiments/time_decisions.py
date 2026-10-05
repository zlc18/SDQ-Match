"""Wall-clock time of one matching decision for every method, measured sequentially in a
single process (no parallel load) on evaluation days 200-202 of the calibration scenario,
and for SDQ-Match on the 40-spot network. Writes results/timing.json."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.env import ScenicEnv  # noqa: E402
from sdq.registry import build, BASELINES, OURS  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402


def ms_per_decision(method, cfg, seeds):
    env = ScenicEnv(cfg)
    total, n = 0.0, 0
    for sd in seeds:
        pol = build(method, cfg)
        obs = env.reset(sd)
        pol.reset(env, sd)
        done = False
        while not done:
            t0 = time.perf_counter()
            target = pol.act(obs)
            total += time.perf_counter() - t0
            n += 1
            obs, _, done, _ = env.step(target)
    return 1000 * total / n


def main():
    out = {"main": {}, "scale_40": {}}
    cfg = get_scenario("main")
    for m in [OURS] + BASELINES:
        out["main"][m] = ms_per_decision(m, cfg, [200, 201, 202])
        print("main", m, round(out["main"][m], 3), flush=True)
    cfg40 = get_scenario("main", S=40, N=80)
    for m in (OURS, "Whittle-F", "PSO-F"):
        out["scale_40"][m] = ms_per_decision(m, cfg40, [200])
        print("scale_40", m, round(out["scale_40"][m], 2), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "timing.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()

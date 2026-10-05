"""Validation-day comparison of the forecaster's behaviour after the pattern test fires:
per-spot ratio rescaling of the history (ratio) against the recently observed per-spot
rate (persist). Validation days 100-119 only. Writes results/check_switch_mode.json."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.env import ScenicEnv  # noqa: E402
from sdq.registry import build  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402

SCEN = {"main": ({}, None), "mismatch_weekend": ({"profile": "weekend"}, "weekday"),
        "mismatch_festival": ({"profile": "festival"}, "weekday"),
        "profile_weekend": ({"profile": "weekend"}, None)}
MODES = [("ratio", 20), ("persist", 20), ("persist", 30), ("persist", 10)]


def job(a):
    tag, method, mode, win, seed = a
    over, hist = SCEN[tag]
    cfg = get_scenario("main", **over)
    hist_cfg = get_scenario("main", **{**over, "profile": hist}) if hist else None
    pol = build(method, cfg, hist_cfg=hist_cfg)
    pol.fc.switch_mode, pol.fc.persist_window = mode, win
    env = ScenicEnv(cfg)
    obs = env.reset(seed)
    pol.reset(env, seed)
    done = False
    while not done:
        obs, _, done, _ = env.step(pol.act(obs))
    return env.metrics()["QASS"]


def main():
    out = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 12) as pool:
        for tag in SCEN:
            for method in ("SDQ-Match", "SDQ-core", "Whittle-F"):
                for mode, win in MODES:
                    q = float(np.mean(pool.map(job, [(tag, method, mode, win, sd)
                                                     for sd in range(100, 120)])))
                    out[f"{tag}/{method}/{mode}{win}"] = q
                    print(tag, method, mode, win, round(q, 2), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                     "check_switch_mode.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

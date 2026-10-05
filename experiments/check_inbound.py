"""Validation-day check of the exact treatment of inbound aircraft (phase 1 split at
landing times) against the earlier approximation (inbound counted as present)."""
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.env import ScenicEnv  # noqa: E402
from sdq.registry import build  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402


def job(a):
    method, over, exact, seed = a
    cfg = get_scenario("main", **over)
    env = ScenicEnv(cfg)
    pol = build(method, cfg)
    core = pol.core if hasattr(pol, "core") else pol
    core.inbound_exact = exact
    obs = env.reset(seed)
    pol.reset(env, seed)
    done = False
    while not done:
        obs, _, done, _ = env.step(pol.act(obs))
    return env.metrics()["QASS"]


def main():
    out = {}
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for tag, over in (("main", {}), ("tau_2", {"tau_rep": 2}), ("tau_8", {"tau_rep": 8}),
                          ("tau_10", {"tau_rep": 10})):
            for method in ("SDQ-Match", "SDQ-core"):
                r = {}
                for exact in (False, True):
                    r[str(exact)] = float(np.mean(pool.map(job, [(method, over, exact, sd)
                                                                  for sd in range(100, 120)])))
                out[f"{tag}/{method}"] = r
                print(tag, method, r, flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results",
                                     "check_inbound.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

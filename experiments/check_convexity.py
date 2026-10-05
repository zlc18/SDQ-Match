"""Numerical check of the convexity assumption behind Proposition 1.

Enumerates the operating grid (initial number in system n0, committed supply,
idle aircraft, phase loads a1 and a2) and tests whether the two-phase transient
cost f(d) is discretely convex in the supply change d; also checks the
steady-state Erlang-A queue length in c. Writes results/convexity.json.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdq.queueing import erlang_a_lq  # noqa: E402
from sdq.scenarios import get_scenario  # noqa: E402
from sdq.transient import TransientCost  # noqa: E402


def main():
    cfg = get_scenario("main")
    tc = TransientCost(cfg.mu, cfg.theta, cfg.tau_rep, 10, cmax=cfg.N, nmax=cfg.N + 45)
    loads = [0.5, 1, 2, 3, 4, 6]
    tot = viol = 0
    for n0, comm, idle, a1, a2 in itertools.product(range(0, 30, 3), range(0, cfg.N - 2),
                                                    range(0, 4), loads, loads):
        if idle > comm:
            continue
        f = tc.delta_costs(n0, comm, idle, a1, a2, cfg.N - comm)
        d2 = np.diff(f, 2)
        tot += 1
        viol += int((d2 < -1e-6 * max(1.0, np.abs(f).max())).any())
    st = st_v = 0
    for a in np.arange(0.2, 9.01, 0.2):
        L = [erlang_a_lq(c, a, cfg.theta / cfg.mu) for c in range(0, cfg.N + 3)]
        st += 1
        st_v += int((np.diff(L, 2) < -1e-9).any())
    out = dict(transient_instances=tot, transient_nonconvex=viol,
               steady_instances=st, steady_nonconvex=st_v)
    print(out)
    path = os.path.join(os.path.dirname(__file__), "..", "results", "convexity.json")
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
